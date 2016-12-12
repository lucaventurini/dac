import java.io.{File, FileWriter, PrintWriter}

import it.polito.dbdmg.ml.L3Ensemble
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Luca Venturini on 23/02/15.
 */
object BacUciCv {

  def main(args: Array[String]) {

    if (args.size < 4) return
    val inputFile = args(0)
    val numModels = args(1)
    val sampleSize = args(2)
    val minSupp = args(3)
    val conf = new SparkConf().setAppName("BAC_UCI_CV_v0.2.0")
    val sc = new SparkContext(conf)






    val all = sc.textFile(inputFile)
    val all2 = {
      if (all.first().startsWith("|")) all.subtract(sc.parallelize(List(all.first()))) else all
    }.filter(_.nonEmpty)
    val data = all2.map(_.split(","))
    val dict = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2, ascending = false).zipWithIndex.collectAsMap
    val transactions = data.map(_.zipWithIndex.map(x => dict(x)))
    val count = transactions.count()
    val numClasses = transactions.map(_.last).max + 1


    val l3 = new L3Ensemble(numModels = numModels.toInt, numClasses = numClasses.toInt, minSupport = minSupp.toDouble, sampleSize = sampleSize.toDouble)






    /* Cross-validation: */
    val numFolds = 10
    val cvTrans = kFold(transactions, numFolds, 12345)
    val measures = cvTrans.map {
      case (train, test) =>
        val t0 = System.nanoTime()
        val model = l3.train(train)
        val t1 = System.nanoTime()
        val labels = test.map(_.find(_ < l3.numClasses)) filter (_.nonEmpty) map (_.get)
        val predictions = model.predict(test.map(_.toSet))

        val t2 = System.nanoTime()
        val confusionMatrix = labels.zip(predictions).groupBy(x => x).mapValues(_.size).collectAsMap() //todo:put a countByKey
        val accuracy = confusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum.toDouble / test.count
        val numRules: Int = model.models.map(_.rules.size).sum
        val numRulesII: Int = model.models.map(_.rulesIIlevel.size).sum
        val numAnts: Int = model.models.map(x => x.rules.map(_.antecedent.size).sum + x.rulesIIlevel.map(_.antecedent.size).sum).sum
        (confusionMatrix,
          numRules,
          numRulesII,
          numAnts,
          (t1 - t0) / 1e6,
          (t2 - t1) / 1e6,
          accuracy)
    }
    //TODO set params
    val cvConfusionMatrix = measures.map(_._1).reduce(
      //(c1, c2) => c1.map{case(k, v) => (k, v+c2(k))}
      (c1, c2) => (c1.keySet ++ c2.keySet).map(k => (k, c1.getOrElse(k, 0) + c2.getOrElse(k, 0))).toMap
    ).mapValues(_.toDouble / count)



    println(s"Confusion Matrix for $inputFile (avg of $numFolds):")
    val accuracies = measures.map(_._7)
    //val accuracy = cvConfusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum
    val accuracy = accuracies.sum / numFolds
    val variance = (accuracies.map(x => math.pow(x - accuracy, 2)).sum -
      math.pow(accuracies.map(_ - accuracy).sum, 2)/numFolds) / numFolds
    println(cvConfusionMatrix.mkString("\n"))
    val numRules: Double = measures.map(_._2).sum / measures.size.toDouble
    val numRulesII: Double = measures.map(_._3).sum / measures.size.toDouble
    val numItemsInRules: Double = measures.map(_._4).sum / measures.size.toDouble
    println("# of rules: " + numRules)
    println("# of II level rules: " + numRulesII)
    println("# of items in all the rules (antecedents): " + numItemsInRules)
    println(s"Accuracy: $accuracy ,variance: $variance")
    val trainTime: Double = measures.map(_._5).sum / measures.size.toDouble
    println("Avg time to generate the model: " + trainTime)
    val predTime: Double = measures.map(_._6).sum / measures.size.toDouble
    println("Avg time to predict: " + predTime)


//    val inf = new File(args(0))
//    println(s"Results of crossvalidation with ${numModels} models, sample size: ${sampleSize}, min supp: ${minSupp}")
//    println(f"${inf.getName}, $accuracy%.4f, $numRules%.4f, $numRulesII%.4f, $numItemsInRules%.4f, $trainTime%.0f, $predTime%.0f, ${accuracies.map(x => f"$x%.4f").mkString(", ")}")









  }

}
