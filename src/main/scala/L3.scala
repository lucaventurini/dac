import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

import scala.util.Try

/**
 * Created by luca on 24/02/15.
 */

case class Rule(antecedent:Set[Long], consequent:Long, support:Double, confidence:Double, chi2:Double) {




  override def toString() = {
    antecedent.mkString(" ") +f" -> $consequent ($support%.6f, $confidence%.6f, $chi2%.6f)"
  }
}

object Rule {

  def ordering[A <: Rule]: Ordering[A] = {
    Ordering.by(x => (x.confidence, x.support, x.antecedent.size, x.toString()))
  }

  implicit def reverseOrdering[A <: Rule]: Ordering[A] = ordering.reverse

}

class L3Model(val dataset:RDD[Array[Long]], var rules:RDD[Rule], val numClasses:Int, val defaultClass:Long, val sort:Boolean = true) extends java.io.Serializable{

  //var rules: List[Rule] = rules.sortBy(x => (x.confidence,x.support,x.antecedent.size,x.antecedent.mkString(", ")), ascending = false).collect().toList//todo: lex order is inverted




  if(sort) rules = rules.sortBy(
    x => (x.confidence,x.support,x.antecedent.size,x.toString()), ascending = false
  )



  def predict(transaction:Set[Long]):Long = {
    val applicable = rules.filter(_.antecedent.subsetOf(transaction))
      if (applicable.isEmpty()){
        defaultClass
      } else applicable.first().consequent

    //rules.find(_.antecedent.subsetOf(transaction)).map(_.consequent).getOrElse(defaultClass)
  }

  /* Beware : RDD do not keep track of order, so do not rely upon order to keep track of a map,
  as in Array[Transaction] -> Array[Prediction]
   */
  def predict(transactions:RDD[Set[Long]]):RDD[(Set[Long], Long)] = {
    val matches = rules.cartesian(transactions).map {
      case (r, t) if(r.antecedent.subsetOf(t))=>(t ->(r, Some(r.consequent)))
      case (r, t) => (t -> (r, None))
    }
     //val predictions =matches .groupByKey().map(_._2.head.getOrElse(defaultClass)) //TODO: groupby does not preserve order!
     matches.groupByKey().map{case (t, l) => l.filter(_._2.nonEmpty) match {
        case x if x.nonEmpty => t -> x.maxBy(_._1)._2.getOrElse(defaultClass)
        case _ => t -> defaultClass
      } }
    //transactions.map(x => predict(x)) //does not work: Spark does not support nested RDDs ops
  }

  def dBCoverage(input: RDD[Array[Long]] = dataset) : L3Model = {
    val usedBuilder = List.newBuilder[Rule] //used rules : correctly predict at least one rule
    val spareBuilder = List.newBuilder[Rule] //spare rules : do not predict, but not harmful
    var db = input.map(_.toSet).collect()
    //db.cache()

    for (r <- rules) {
      val applicable = db.filter(x => r.antecedent.subsetOf(x))
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


    new L3Model(dataset, input.sparkContext.parallelize(usedBuilder.result ::: spareBuilder.result), numClasses, defaultClass, sort=false)

  }

  override def toString() = {
    rules.map(_.toString).collect().mkString("\n")
  }

}

class L3EnsembleModel(val models:Array[L3Model]) {

  def predict(transaction:Set[Long]):Long = {
    /* use majority voting to select a prediction */
    models.map(_.predict(transaction)).groupBy{label => label }.mapValues(_.size).maxBy(_._2)._1
  }

  def predict(transactions:RDD[Set[Long]]):RDD[Long] = {
    transactions.map(x => predict(x)) //todo: switch to models.map(_.predict(transactions)).majority_voting
  }

  override def toString() = {
    models.zipWithIndex.map{case (m ,i) => s"Model $i:\n${m}\n"}.mkString
  }

}

class L3Ensemble (val numClasses:Int, val numModels:Int = 100, val minSupport:Double = 0.2, val minConfidence:Double = 0.5, val minChi2:Double = 3.841){

  def train(input: RDD[Array[Long]]):L3EnsembleModel = {
    val l3 = new L3(numClasses, minSupport, minConfidence, minChi2)
    val models = Array.fill(numModels)(scala.util.Random.nextLong()).
      map(input.sample(true, 0.01, _)). //todo: variable sample size
      map(x => l3.train(x))
    new L3EnsembleModel(models)

  }
}

class L3 (val numClasses:Int, val minSupport:Double = 0.2, val minConfidence:Double = 0.5, val minChi2:Double = 3.841) extends java.io.Serializable{

  def train(input: RDD[Array[Long]]): L3Model = {

    val count = input.count()

    val fpg = new FPGrowth()
      .setMinSupport(minSupport)
      .setNumPartitions(10) //TODO
    val model = fpg.run(input)


    //model.freqItemsets.map{case (items, sup) => (items.partition(_ < numClasses), sup)}

    val antecedents = model.freqItemsets.map{
      f => val x = f.items.partition(_ < numClasses);(x._2.toSet,(x._1, f.freq.toDouble/count))
    }
    val supAnts = antecedents.filter(_._2._1.isEmpty).map{
      case (ant, (_, sup)) => (ant, sup)
    }

    val supClasses = antecedents.filter(_._1.isEmpty).map{
      case (_, (classLabels, sup)) => (classLabels(0), sup)
    }.collectAsMap()


    val rules = antecedents.filter(_._2._1.nonEmpty).join(supAnts).map{
      case (ant, ((classLabels, sup), supAnt)) => {
        val supCons = supClasses(classLabels(0))
        val chi2 = count * {List(sup, supAnt - sup, supCons - sup, 1 - supAnt - supCons + sup) zip
          List(supAnt * supCons, supAnt * (1 - supCons), (1 - supAnt) * supCons, (1 - supAnt) * (1 - supCons)) map
          {case (observed, expected) => math.pow((observed - expected), 2) / expected} sum }
        Rule(ant, classLabels(0), sup, sup/supAnt.toDouble, chi2)
      }
    }.filter(r => r.confidence >= minConfidence && (r.chi2 >= minChi2 || r.chi2.isNaN))

    new L3Model(input, rules, numClasses, supClasses.maxBy(_._2)._1)
  }

  def train[Item](input: RDD[Array[Item]], isClassLabel:(Item => Boolean)) {
    //TODO: generalize to any type of item representation, given a function to distinguish a class label
    //default: Item = Long, isClassLabel = (x => x < numClasses)
  }

}
