package it.polito.dbdmg.ml

import it.polito.dbdmg.spark.mllib.fpm.{FPGrowth => FPGrowthLocal}
import org.apache.spark.SparkException
import org.apache.spark.mllib.fpm.{FPGrowth => PFP}
import org.apache.spark.rdd.RDD

import scala.util.Random
import scala.collection.mutable.HashMap


/**
  * Created by luca on 24/02/15.
  */

object Rule {
  def orderingByConf[A <: Rule[Long]]: Ordering[A] =
    Ordering.by(x => (x.confidence,x.support,x.antecedent.size,x.toString()))

  def orderingByConfString[A <: Rule[String]]: Ordering[A] =
    Ordering.by(x => (x.confidence,x.support,x.antecedent.size,x.toString()))

  implicit def orderingByConfDesc[A <: Rule[Long]]: Ordering[A] =
    orderingByConf[A].reverse

  implicit def orderingByConfDescString[A <: Rule[String]]: Ordering[A] =
    orderingByConfString[A].reverse

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

  def appliesTo(x : Iterable[T]) : Boolean = {
    antecedent.forall(a => x.exists(_==a))
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

class L3LocalModel(val rules:List[Rule[Long]],
                   val rulesIIlevel:List[Rule[Long]],
                   val numClasses:Int,
                   val defaultClass:Long,
                   val classes:Array[Long]=Array(0L,1L)) extends java.io.Serializable {

//  lazy val classes = (rules).map(_.consequent).distinct.toArray //taking classes from 1st level only should work, no?

  @deprecated def predict(transaction:Set[Long]):Long = {
    //sortedRules.filter(_.antecedent.subsetOf(transaction)).first().consequent
    predictOption(transaction).getOrElse(defaultClass)
  }
  @deprecated def predictOption(transaction:Set[Long]):Option[Long] = {
    rules.find(_.appliesTo(transaction)).map(_.consequent).
      orElse(rulesIIlevel.find(_.appliesTo(transaction)).map(_.consequent))
  }

  def predict[T<:Iterable[Long]](transaction:T, withWeight:Boolean = false):Long = {
    //sortedRules.filter(_.antecedent.subsetOf(transaction)).first().consequent
    predictOption(transaction).getOrElse(defaultClass)
  }
  def predictOption[T<:Iterable[Long]](transaction:T, withWeight:Boolean = false):Option[Long] = {
    if (withWeight) {
      //use predictProba
      val votes = classes zip predictProba(transaction)
        if (votes.isEmpty)
          None
        else
          Some(votes.maxBy(_._2)._1)

    } else {
      //classic prediction: first match
      rules.find(_.appliesTo(transaction)).map(_.consequent).
        orElse(rulesIIlevel.find(_.appliesTo(transaction)).map(_.consequent))
    }
  }

  def predict[T<:Iterable[Long]](transactions:RDD[T]):RDD[Long] = {
      transactions.map(x => predict(x))
  }

  def predictProba[T<:Iterable[Long]](transaction:T):Array[Double] = {
    val votes = rules.filter(_.appliesTo(transaction))
      .groupBy(_.consequent)
//      .mapValues(_.map(_.confidence)) // take confidence as weight
      .mapValues(_.map(_.support)) // take support as weight
//      .mapValues(weights => 1-weights.map(1-_).product)  //voting function: 1-PROD(1-p_i)
      .mapValues(weights => weights.max)  //voting function: MAX
    //todo: use II level rules
    val probaRest = votes.map(1 - _._2).product //PROD(1-p_i)
    val probas = classes.map(votes.getOrElse(_, probaRest))
    val sumProba = probas.sum

    probas.map(_ / sumProba)
  }

  def predictProba[T<:Iterable[Long]](transactions:RDD[T]):RDD[Array[Double]] = {
      transactions.map(x => predictProba(x))
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

  //define how to sum 2 or more rules here (for model reduction)
  def sumRules(rules: List[Rule[Long]]): Rule[Long] = {
    val conf = 1-rules.map(1-_.confidence).product //new conf = 1-PROD(1-conf_i)
    val sup = rules.map(_.support).max // new support = max sup_i
    val chi2 = rules.map(_.chi2).max
    new Rule(rules(0).antecedent, rules(0).consequent, support = sup, confidence = conf, chi2 = chi2)
  }

  def merge(other: L3LocalModel):L3LocalModel = {
    val rules = (this.rules++other.rules).groupBy(x => (x.antecedent.toSet, x.consequent)).mapValues(sumRules(_)).values
    val rulesII = (this.rulesIIlevel++other.rulesIIlevel).groupBy(x => (x.antecedent.toSet, x.consequent)).mapValues(sumRules(_)).values
    new L3LocalModel(rules.toList, rulesII.toList, numClasses = numClasses, classes = classes, defaultClass = defaultClass)
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


  def predict[T<:Iterable[Long]](transaction:T):Long = {
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

class L3EnsembleModel(val models:Array[L3LocalModel],
                      val preferredClass:Option[Long]=None,
                      var withWeight:Boolean = false) extends java.io.Serializable {

  // choose default class by majority, if preferred is not specified
  lazy val defaultClass : Long = {
    preferredClass.getOrElse(models.map(_.defaultClass).groupBy{label => label }.mapValues(_.size).maxBy(_._2)._1)
  }

  def lightModel:L3LocalModel = {
    models.reduce(_.merge(_))
  }

  def setWithWeight(value: Boolean) = {
    withWeight = value
  }

  def isWithWeight:Boolean = {
    withWeight
  }

  def predict[T<:Iterable[Long]](transaction:T):Long = {
    /* use majority voting to select a prediction */
    predictMajorityOption(transaction)
  }
  def predictProba[T<:Iterable[Long]](transaction:T):Array[Double] = {
//    val probas = models.map(_.predictProba(transaction).map(1-_)).reduce((a, b) => (a zip b).map(x => x._1*x._2)).map(1-_) //1 - PROD (1-p_i)
//    val probas = models.map(_.predictProba(transaction)).reduce((a, b) => (a zip b).map(x => x._1*x._2)) //PROD (p_i)
    val probas = models.map(_.predictProba(transaction)).reduce((a, b) => (a zip b).map(x => x._1+x._2)) //SUM (p_i)
    val sum = probas.sum
    probas.map(_ / sum) //normalize
  }

  def predictProba[T<:Iterable[Long]](transactions:RDD[T]):RDD[Array[Double]] = {
    transactions.map(x => predictProba(x))
  }
  def predict[T<:Iterable[Long]](transactions:RDD[T]):RDD[Long] = {
    transactions.map(x => predict(x))
  }

  def predictMajority[T<:Iterable[Long]](transaction:T):Long = {
    /* use majority voting to select a prediction */
    models.map(_.predict(transaction, withWeight)).groupBy{label => label }.mapValues(_.size).maxBy(_._2)._1
  }

  def predictMajorityOption[T<:Iterable[Long]](transaction:T):Long = {
    /* use majority voting to select a prediction */
    val votes = models.map(_.predictOption(transaction, withWeight = withWeight)).filter(_.nonEmpty).groupBy{label => label }.mapValues(_.size)
    if (votes.nonEmpty) votes.maxBy(_._2)._1.getOrElse(defaultClass) else defaultClass
  }

  def dbCoverage(dataset:Iterable[(Array[Long], Long)], saveSpare:Boolean = true) = {
    new L3EnsembleModel(models.map(_.dBCoverage(dataset, saveSpare = saveSpare)))
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
                  val strategy: String = "support",
                  val minInfoGain: Double = 0.0,
                  val rulesMaxLen: Int = 10,
                  val preferredClass: Option[Long] = None,
                  val withDBCoverage:Boolean = true,
                  val saveSpare:Boolean = true,
                  val withShuffling:Boolean = true,
                  val withReplacement:Boolean = true) extends java.io.Serializable{

  def train(input: RDD[(Array[Long], Long)]):L3EnsembleModel = {
    // N.B: numPartitions = numModels
    if (!(strategy.equals("gain") || strategy.equals("support")))
      throw new SparkException(s"Strategy must be gain or support but got: $strategy.")
    if (!withShuffling && withReplacement)
      throw new SparkException(s"Sampling with replacement without shuffling is deprecated. Aborting.")
    if (!withShuffling && input.partitions.length < numModels)
      throw new SparkException(s"Cannot generate $numModels models with only ${input.partitions.length} partitions. Aborting.")
    val l3 = new L3(numClasses, minSupport, minConfidence, minChi2, strategy)
    val partitions: RDD[(Array[Long], Long)] = withShuffling match {
      case true => input.keyBy(_ => Random.nextInt()).
      sample(withReplacement, numModels * sampleSize). //TODO: stratified sampling
      repartition(numModels).values
      case false => input.sample(false, numModels * sampleSize).coalesce(numModels)}
    new L3EnsembleModel(
      partitions.
        mapPartitions{ samples =>
          val s = samples.toIterable //todo: toArray ?
        val model: L3LocalModel = {
          if (withDBCoverage) l3.train(s).dBCoverage(s, saveSpare=saveSpare)
          else l3.train(s)
        }
          Iterator(model)
        }.collect(),
      preferredClass
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
          val strategy: String = "support",
          val minInfoGain: Double = 0.0,
          val rulesMaxLen: Int = 10) extends java.io.Serializable{

  def train(input: Iterable[(Array[Long], Long)]): L3LocalModel = {
    if (!(strategy.equals("gain") || strategy.equals("support")))
      throw new SparkException(s"Strategy must be gain or support but got: $strategy.")
    def countByValue[T](xs: TraversableOnce[T]): Map[T, Int] = {
      xs.foldLeft(HashMap.empty[T, Int].withDefaultValue(0))((acc, x) => { acc(x) += 1; acc}).toMap
    }
    val classCount = countByValue(input.map(_._2).toIterator)//.groupBy(x => x).mapValues(_.size)
    val defaultClass = classCount.maxBy(_._2)._1
    val fpg = new FPGrowthLocal[Long]()
      .setMinSupport(minSupport)
      .setStrategy(strategy)
      .setMinConfidence(minConfidence)
      .setMinChi2(minChi2)
      .setMinInfoGain(minInfoGain)
      .setRulesMaxLength(rulesMaxLen)

    val rules = fpg.run(input, classCount)

    new L3LocalModel(rules.toList.sorted, List(), numClasses, defaultClass)

  }



  /*
   * This version uses PFP (Parallel FP-Growth) and doesn't have the class
   * information in building FP-Tree so cannot extract rules directly.
   * Obsolete
   */
  @deprecated def train(input: RDD[Array[Long]]): L3Model = {

    val count = input.count()

    val fpg = new PFP()
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

  def withInformationGain(): Boolean = {
    if (strategy.equals("gain")) true else false
  }

}
