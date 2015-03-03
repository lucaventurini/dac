import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.util.Try

/**
 * Created by luca on 24/02/15.
 */

case class Rule(antecedent:Set[Long], consequent:Long, support:Double, confidence:Double, chi2:Double) {
  override def toString() = {
    antecedent.mkString(" ") +f" -> $consequent ($support%.6f, $confidence%.6f, $chi2%.6f)"
  }
}

class L3Model(val rules:RDD[Rule], val numClasses:Int) extends java.io.Serializable{

  lazy val sortedRules = rules.sortBy(_.antecedent.size).collect()//todo: lex order

  def predict(transaction:Set[Long]):Option[Long] = {
    //sortedRules.filter(_.antecedent.subsetOf(transaction)).first().consequent
    sortedRules.find(_.antecedent.subsetOf(transaction)).map(_.consequent)
  }

  def predict(transactions:RDD[Set[Long]]):RDD[Option[Long]] = {
    transactions.map(x => predict(x)) //does not work: Spark does not support nested RDDs ops
  }

  def dBCoverage(input: RDD[Array[Long]]) : L3Model = {
    val usedBuilder = List.newBuilder[Rule] //used rules : correctly predict at least one rule
    val spareBuilder = List.newBuilder[Rule] //spare rules : do not predict, but not harmful
    var db = input.map(_.toSet)
    for (r <- sortedRules) {
      val applicable = db.filter(x => r.antecedent.subsetOf(x))
      if (applicable.isEmpty()) {
        spareBuilder += r
      }
      else {
        val correct = applicable.filter {
          x => val classLabel = x.find(_ < numClasses); classLabel == this.predict(x)
        }
        if (!correct.isEmpty()) {
          db = db.subtract(applicable)
          usedBuilder += r
        }
      }
    }



    this
  }

  override def toString() = {
    rules.collect().map(_.toString).mkString("\n")
  }
}

class L3EnsembleModel(val models:Array[L3Model]) {

  def predict(transaction:Set[Long]):Option[Long] = {
    /* use majority voting to select a prediction */
    Try(models.map(_.predict(transaction)).groupBy{case Some(label) => label }.mapValues(_.size).maxBy(_._2)._1).toOption //todo: try is non efficient
  }
}

class L3Ensemble (val numClasses:Int, val numModels:Int = 100, val minSupport:Double = 0.2, val minConfidence:Double = 0.5, val maxChi2:Double = 3.841){

  def train(input: RDD[Array[Long]]):L3EnsembleModel = {
    val l3 = new L3(numClasses, minSupport, minConfidence, maxChi2)
    val models = Array.fill(numModels)(scala.util.Random.nextLong()).
      map(input.sample(true, 0.01, _)). //todo: variable sample size
      map(x => l3.train(x))
    new L3EnsembleModel(models)

  }
}

class L3 (val numClasses:Int, val minSupport:Double = 0.2, val minConfidence:Double = 0.5, val maxChi2:Double = 3.841) extends java.io.Serializable{

  def train(input: RDD[Array[Long]]): L3Model = {

    val count = input.count()

    val fpg = new FPGrowth()
      .setMinSupport(minSupport)
      .setNumPartitions(10) //TODO
    val model = fpg.run(input)


    //model.freqItemsets.map{case (items, sup) => (items.partition(_ < numClasses), sup)}

    val antecedents = model.freqItemsets.map{
      case (items, freq) => val x = items.partition(_ < numClasses);(x._2.toSet,(x._1, freq.toDouble/count))
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
    }.filter(r => r.confidence >= minConfidence && r.chi2 <= maxChi2)

    new L3Model(rules)
  }

  def train[Item](input: RDD[Array[Item]], isClassLabel:(Item => Boolean)) {
    //TODO: generalize to any type of item representation, given a function to distinguish a class label
    //default: Item = Long, isClassLabel = (x => x < numClasses)
  }

}
