package it.polito.dbdmg.ml

import it.polito.dbdmg.spark.mllib.fpm.{FPGrowth => FPGrowthLocal}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * Created by luca on 24/02/15.
 */

object Rule {
  def orderingByConf[A <: Rule]: Ordering[A] =
    Ordering.by(x => (x.confidence,x.support,x.antecedent.size,x.toString()))

  implicit def orderingByConfDesc[A <: Rule]: Ordering[A] =
    orderingByConf.reverse



}

case class Rule(antecedent:Array[Long], consequent:Long, support:Double, confidence:Double, chi2:Double) {
  override def toString() = {
    antecedent.mkString(" ") +f" -> $consequent ($support%.6f, $confidence%.6f, $chi2%.6f)"
  }

  def appliesTo(x : Array[Long]) : Boolean = {
    antecedent.forall(x.contains(_))
  }

  @deprecated def appliesTo(x : Set[Long]) : Boolean = {
    antecedent.forall(x.contains(_))
  }
}

class L3LocalModel(val rules:List[Rule], val rulesIIlevel:List[Rule], val numClasses:Int, val defaultClass:Long) extends java.io.Serializable {

  def predict(transaction:Set[Long]):Long = {
    //sortedRules.filter(_.antecedent.subsetOf(transaction)).first().consequent
    predictOption(transaction).getOrElse(defaultClass)
  }
  def predictOption(transaction:Set[Long]):Option[Long] = {
    rules.find(_.appliesTo(transaction)).map(_.consequent).
      orElse(rulesIIlevel.find(_.appliesTo(transaction)).map(_.consequent))
  }

  def predict(transactions:RDD[Set[Long]]):RDD[Long] = {
    transactions.map(x => predict(x))
  }

  def dBCoverage(input: Iterable[Array[Long]], saveSpare: Boolean = true) : L3LocalModel = {
    val usedBuilder = List.newBuilder[Rule] //used rules : correctly predict at least one rule
    val spareBuilder = List.newBuilder[Rule] //spare rules : do not predict, but not harmful
    var db = input.toSeq
    //db.cache()

    for (r <- rules) {
      val applicable = db.filter(x => r.appliesTo(x))
      if (applicable.isEmpty) {
        if (saveSpare)spareBuilder += r
      }
      else {
        val correct = applicable.find {
          x => val classLabel = x.find(_ < numClasses)
            classLabel == Some(r.consequent)
        }
        if (correct.nonEmpty) {
          db = db diff applicable

          usedBuilder += r
        }
      }
    }

    new L3LocalModel(usedBuilder.result, spareBuilder.result, numClasses, defaultClass)

  }

  override def toString() = {
    (rules ++ rulesIIlevel).map(_.toString).mkString("\n")
  }


}

class L3Model(val dataset:RDD[Array[Long]], val rules:List[Rule], val numClasses:Int, val defaultClass:Long) extends java.io.Serializable{ //todo: serialize with Kryo?

  //var rules: List[Rule] = rules.sortBy(x => (x.confidence,x.support,x.antecedent.size,x.antecedent.mkString(", ")), ascending = false).collect().toList//todo: lex order is inverted

  def this(dataset:RDD[Array[Long]], rules:RDD[Rule], numClasses:Int, defaultClass:Long) {
    // this constructor SORTS the rules in input, while the default one preserves the order of the list
    this(dataset,
      rules.sortBy(
        x => (x.confidence,x.support,x.antecedent.size,x.toString()), ascending = false
      ).collect().toList,
      numClasses,
      defaultClass)
  }

  def predict(transaction:Set[Long]):Long = {
    //sortedRules.filter(_.antecedent.subsetOf(transaction)).first().consequent
    rules.find(_.appliesTo(transaction)).map(_.consequent).getOrElse(defaultClass)
  }

  def predict(transactions:RDD[Set[Long]]):RDD[Long] = {
    transactions.map(x => predict(x)) //does not work: Spark does not support nested RDDs ops
  }

  def dBCoverage(input: RDD[Array[Long]] = dataset) : L3Model = {
    val usedBuilder = List.newBuilder[Rule] //used rules : correctly predict at least one rule
    val spareBuilder = List.newBuilder[Rule] //spare rules : do not predict, but not harmful
    var db = input.collect()
    //db.cache()

    for (r <- rules) {
      val applicable = db.filter(x => r.appliesTo(x))
      if (applicable.isEmpty) {
        spareBuilder += r
      }
      else {
        val correct = applicable.filter {
          x => val classLabel = x.find(_ < numClasses)
            classLabel == Some(r.consequent)
        }
        if (correct.nonEmpty) {
          db = db diff applicable

          usedBuilder += r
        }
      }
    }

    new L3Model(dataset, usedBuilder.result ::: spareBuilder.result, numClasses, defaultClass)

  }

  override def toString() = {
    rules.map(_.toString).mkString("\n")
  }

}

class L3EnsembleModel(val models:Array[L3LocalModel]) extends java.io.Serializable {

  // choose default class by majority
  lazy val defaultClass : Long = models.map(_.defaultClass).groupBy{label => label }.mapValues(_.size).maxBy(_._2)._1

  def predict(transaction:Set[Long]):Long = {
    /* use majority voting to select a prediction */
    predictMajorityOption(transaction)
  }

  def predict(transactions:RDD[Set[Long]]):RDD[Long] = {
    transactions.map(x => predict(x)) //todo: switch to models.map(_.predict(transactions)).majority_voting
  }

  def predictMajority(transaction:Set[Long]):Long = {
    /* use majority voting to select a prediction */
    models.map(_.predict(transaction)).groupBy{label => label }.mapValues(_.size).maxBy(_._2)._1
  }

  def predictMajorityOption(transaction:Set[Long]):Long = {
    /* use majority voting to select a prediction */
    val votes = models.map(_.predictOption(transaction)).filter(_.nonEmpty).groupBy{label => label }.mapValues(_.size)
    if (votes.nonEmpty) votes.maxBy(_._2)._1.getOrElse(defaultClass) else defaultClass
  }

  def dbCoverage(dataset:Iterable[Array[Long]]) = {
    new L3EnsembleModel(models.map(_.dBCoverage(dataset)))
  }

  override def toString() = {
    models.zipWithIndex.map{case (m ,i) => s"Model $i:\n${m}\n"}.mkString
  }

}

class L3Ensemble (val numClasses:Int,
                  val numModels:Int = 100,
                  val sampleSize:Double = 0.01,
                  val minSupport:Double = 0.2,
                  val minConfidence:Double = 0.5,
                  val minChi2:Double = 3.841,
                  val withReplacement:Boolean = true) extends java.io.Serializable{

  def train(input: RDD[Array[Long]]):L3EnsembleModel = {
    // N.B: numPartitions = numModels
    val l3 = new L3(numClasses, minSupport, minConfidence, minChi2)
    new L3EnsembleModel(
      input.keyBy(_ => Random.nextInt()).
        sample(withReplacement, numModels*sampleSize).//TODO: stratified sampling
        repartition(numModels).
        mapPartitions{ samples =>
        val s = samples.toIterable.map(_._2) //todo: toArray ?
        val model: L3LocalModel = l3.train(s).dBCoverage(s)
        Iterator(model)
      }.collect()
    )
  }


}

class L3 (val numClasses:Int, val minSupport:Double = 0.2, val minConfidence:Double = 0.5, val minChi2:Double = 3.841) extends java.io.Serializable{

  def train(input: Iterable[Array[Long]]): L3LocalModel = {
    val count = input.size

    val fpg = new FPGrowthLocal()
      .setMinSupport(minSupport)
    val model = fpg.run(input)


    //model.freqItemsets.map{case (items, sup) => (items.partition(_ < numClasses), sup)}

    val antecedents = model.freqItemsets.map{
      f => val x = f.items.partition(_ < numClasses);(x._2.toSet,(x._1, f.freq.toDouble/count))
    }.toList
    val supAnts = antecedents.filter(_._2._1.isEmpty).map{
      case (x, (_, sup)) => (x, sup)
    }.toMap

    val supClasses = antecedents.filter(_._1.isEmpty).map{
      case (_, (classLabels, sup)) => (classLabels(0), sup)
    }.toMap





    val rules = antecedents.filter(x =>x._2._1.nonEmpty && x._1.nonEmpty).map{
      case (ant, (classLabels, sup)) => {
        val supCons = supClasses(classLabels(0))
        val supAnt = supAnts(ant)
        val chi2 = count * {List(sup, supAnt - sup, supCons - sup, 1 - supAnt - supCons + sup) zip
          List(supAnt * supCons, supAnt * (1 - supCons), (1 - supAnt) * supCons, (1 - supAnt) * (1 - supCons)) map
          {case (observed, expected) => math.pow((observed - expected), 2) / expected} sum }
        Rule(ant.toArray, classLabels(0), sup, sup/supAnt.toDouble, chi2) //todo: stop using sets
      }
    }.filter(r => r.confidence >= minConfidence && (r.chi2 >= minChi2 || r.chi2.isNaN))


    new L3LocalModel(rules.sorted, List(), numClasses, supClasses.maxBy(_._2)._1)

  }

  def train(input: RDD[Array[Long]]): L3Model = {

    val count = input.count()

    val fpg = new FPGrowth()
      .setMinSupport(minSupport)
      .setNumPartitions(4*2) //TODO
    val model = fpg.run(input)


    //model.freqItemsets.map{case (items, sup) => (items.partition(_ < numClasses), sup)}

    val antecedents = model.freqItemsets.map{
      f => val x = f.items.partition(_ < numClasses);(x._2.toSet,(x._1, f.freq.toDouble/count))
    }
    antecedents.cache()
    val supAnts = antecedents.filter(_._2._1.isEmpty).mapValues{
      case (_, sup) => sup
    }

    val supClasses = antecedents.lookup(Set()).map{
      case (classLabels, sup) => (classLabels(0), sup)
    }.toMap //todo:bottleneck
    //1st: use broadcast instead of a local map (sends the map only once)




    val rules = antecedents.filter(_._2._1.nonEmpty).join(supAnts).map{
      case (ant, ((classLabels, sup), supAnt)) => {
        val supCons = supClasses(classLabels(0))
        val chi2 = count * {List(sup, supAnt - sup, supCons - sup, 1 - supAnt - supCons + sup) zip
          List(supAnt * supCons, supAnt * (1 - supCons), (1 - supAnt) * supCons, (1 - supAnt) * (1 - supCons)) map
          {case (observed, expected) => math.pow((observed - expected), 2) / expected} sum }
        Rule(ant.toArray, classLabels(0), sup, sup/supAnt.toDouble, chi2) //todo: stop using sets
      }
    }.filter(r => r.confidence >= minConfidence && (r.chi2 >= minChi2 || r.chi2.isNaN))

    new L3Model(input, rules, numClasses, supClasses.maxBy(_._2)._1)
  }

  def train[Item](input: RDD[Array[Item]], isClassLabel:(Item => Boolean)) {
    //TODO: generalize to any type of item representation, given a function to distinguish a class label
    //default: Item = Long, isClassLabel = (x => x < numClasses)
  }

}
