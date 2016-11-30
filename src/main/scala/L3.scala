package it.polito.dbdmg.ml

import it.polito.dbdmg.spark.mllib.fpm.{FPGrowth => FPGrowthLocal}
import org.apache.spark.{SparkContext, SparkException}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * Created by luca on 24/02/15.
  */

object Rule {
  def orderingByConf[A <: Rule[Long]]: Ordering[A] =
    Ordering.by(x => (x.confidence,x.support,x.antecedent.size,x.toString()))

  implicit def orderingByConfDesc[A <: Rule[Long]]: Ordering[A] =
    orderingByConf.reverse

}

case class Rule[T](antecedent:Array[T], consequent:T, support:Double, confidence:Double, chi2:Double) {

  override def toString() = {
    antecedent.mkString(" ") + f" -> $consequent ($support%.6f, $confidence%.6f, $chi2%.6f)"
  }

  override def equals(obj: scala.Any): Boolean = {
    val r: Rule[T] = obj match {
      case r: Rule[T] => r
      case _ => return false
    }
    if (! antecedent.sameElements(r.antecedent)) return false
    if (consequent != r.consequent) return false
    return true
  }


  def appliesTo(x : Array[T]) : Boolean = {
    antecedent.forall(x.contains(_))
  }

  def appliesTo(x : (Array[T], T)) : Boolean = {
    antecedent.forall(x._1.contains(_))
  }

  @deprecated def appliesTo(x : Set[T]) : Boolean = {
    antecedent.forall(x.contains(_))
  }

  def printAntecedent() = {
    antecedent.mkString(" ")
  }
}

class L3LocalModel(val rules:List[Rule[Long]], val rulesIIlevel:List[Rule[Long]], val numClasses:Int, val defaultClass:Long) extends java.io.Serializable {

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

  def dBCoverage(input: Iterable[(Array[Long], Long)], saveSpare: Boolean = true) : L3LocalModel = {
    val usedBuilder = List.newBuilder[Rule[Long]] //used rules : correctly predict at least one rule
    val spareBuilder = List.newBuilder[Rule[Long]] //spare rules : do not predict, but not harmful
    var db = input.toSeq
    //db.cache()

    for (r <- rules) {
      val applicable = db.filter(x => r.appliesTo(x))
      if (applicable.isEmpty) {
        if (saveSpare) spareBuilder += r
      }
      else {
        val correct = applicable.find {
          x => val classLabel = x._2
            classLabel == r.consequent
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

  def avgItemsInRules = {
    val ants = rules.map(_.antecedent.size).sum
    val size: Double = rules.size.toDouble
    if (ants == 0 || size == 0.0)  0.0 else ants / size
  }

  def avgItemsInRulesII = {
    val ants = rulesIIlevel.map(_.antecedent.size).sum
    val size: Double = rulesIIlevel.size.toDouble
    if (ants == 0 || size == 0.0)  0.0 else ants / size
  }


}

class L3Model(val dataset:RDD[Array[Long]], val rules:List[Rule[Long]], val numClasses:Int, val defaultClass:Long) extends java.io.Serializable{ //todo: serialize with Kryo?

  //var rules: List[Rule] = rules.sortBy(x => (x.confidence,x.support,x.antecedent.size,x.antecedent.mkString(", ")), ascending = false).collect().toList//todo: lex order is inverted


  def this(dataset:RDD[Array[Long]], rules:RDD[Rule[Long]], numClasses:Int, defaultClass:Long) {
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
    val usedBuilder = List.newBuilder[Rule[Long]] //used rules : correctly predict at least one rule
    val spareBuilder = List.newBuilder[Rule[Long]] //spare rules : do not predict, but not harmful
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

  def dbCoverage(dataset:Iterable[(Array[Long], Long)]) = {
    new L3EnsembleModel(models.map(_.dBCoverage(dataset)))
  }

  override def toString() = {
    models.zipWithIndex.map{case (m ,i) => s"Model $i:\n${m}\n"}.mkString
  }

  def totRules = {
    models.map(_.rules.size).sum
  }

  def totRulesII = {
    models.map(_.rulesIIlevel.size).sum
  }

  def avgItemsInRules = {
    val ants = models.map(_.avgItemsInRules).sum
    val size: Double = models.size.toDouble
    if (ants == 0 || size == 0.0)  0.0 else ants / size
  }

  def avgItemsInRulesII = {
    val ants = models.map(_.avgItemsInRulesII).sum
    val size: Double = models.size.toDouble
    if (ants == 0 || size == 0.0)  0.0 else ants / size
  }

}

class L3Ensemble (val numClasses:Int,
                  val numModels:Int = 100,
                  val sampleSize:Double = 0.01,
                  val minSupport:Double = 0.2,
                  val minConfidence:Double = 0.5,
                  val minChi2:Double = 3.841,
                  val strategy: String = "gain",
                  val withReplacement:Boolean = true) extends java.io.Serializable{

  def train(input: RDD[(Array[Long], Long)]):L3EnsembleModel = {
    // N.B: numPartitions = numModels
    if (!(strategy.equals("gain") || strategy.equals("support")))
      throw new SparkException(s"Strategy must be gain or support but got: $strategy.")
    val l3 = new L3(numClasses, minSupport, minConfidence, minChi2, strategy)
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

  def withInformationGain(): Boolean = {
    if (strategy.equals("gain")) true else false
  }


}

class L3 (val numClasses:Int,
          val minSupport:Double = 0.2,
          val minConfidence:Double = 0.5,
          val minChi2:Double = 3.841,
          val strategy: String = "gain") extends java.io.Serializable{

  def train(input: Iterable[(Array[Long], Long)]): L3LocalModel = {
    if (!(strategy.equals("gain") || strategy.equals("support")))
      throw new SparkException(s"Strategy must be gain or support but got: $strategy.")
    val classCount = input.map(_._2).groupBy(x => x).mapValues(_.size)
    val defaultClass = classCount.maxBy(_._2)._1
    val fpg = new FPGrowthLocal[Long]()
      .setMinSupport(minSupport)
      .setStrategy(strategy)
      .setMinConfidence(minConfidence)
      .setMinChi2(minChi2)

    val rules = fpg.run(input, classCount)

    new L3LocalModel(rules.toList.sorted, List(), numClasses, defaultClass)

  }

  def withInformationGain(): Boolean = {
    if (strategy.equals("gain")) true else false
  }

}
