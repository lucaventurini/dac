import java.io.{File, FileWriter, PrintWriter}

import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Luca Venturini on 23/02/15.
 */
object L3uciTestsBagging {

  def main(args: Array[String]) {
    val inputFolder = "/home/lucav/data/UCI/test5/"
    //val inputFile = "/home/lucav/data/UCI/test1/voting.data" // Should be some file on your system
    if (args.size < 4) return
    val inputFile = args(0)
    val numModels = args(1)
    val sampleSize = args(2)
    val minSupp = args(3)
    val conf = new SparkConf().setAppName("L3Local_UCI_v0.2.0").setMaster("local[*]")
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


    val l3 = new L3Ensemble(numModels = numModels.toInt, numClasses = numClasses.toInt, minSupport = minSupp.toDouble)






    /* Cross-validation: */
    val numFolds = 10
    val cvTrans = kFold(transactions, numFolds, 12345)
    val measures = cvTrans.map {
      case (train, test) =>
        val t0 = System.nanoTime()
        val model = l3.train(train).dbCoverage()
        val t1 = System.nanoTime()
        val labels = test.map(_.find(_ < l3.numClasses)) filter (_.nonEmpty) map (_.get)
        val predictions = model.predict(test.map(_.toSet))
        //todo: should we remove the class labels?
        val t2 = System.nanoTime()
        val confusionMatrix = labels.zip(predictions).groupBy(x => x).mapValues(_.size).collectAsMap() //todo:put a countByKey
        val accuracy = confusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum.toDouble / test.count
        (confusionMatrix, model.models.map(_.rules.size).sum, (t1 - t0) / 1e6, (t2 - t1) / 1e6, accuracy)
    }
    //TODO set params
    val cvConfusionMatrix = measures.map(_._1).reduce(
      //(c1, c2) => c1.map{case(k, v) => (k, v+c2(k))}
      (c1, c2) => (c1.keySet ++ c2.keySet).map(k => (k, c1.getOrElse(k, 0) + c2.getOrElse(k, 0))).toMap
    ).mapValues(_.toDouble / count)



    println(s"Confusion Matrix for $inputFile (avg of $numFolds):")
    val accuracies = measures.map(_._5)
    //val accuracy = cvConfusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum
    val accuracy = accuracies.sum / numFolds
    val variance = (accuracies.map(x => math.pow(x - accuracy, 2)).sum -
      math.pow(accuracies.map(_ - accuracy).sum, 2)/numFolds) / numFolds
    println(cvConfusionMatrix.mkString("\n"))
    val numRules: Double = measures.map(_._2).sum / measures.size.toDouble
    println("# of rules: " + numRules)
    println(s"Accuracy: $accuracy ,variance: $variance")
    val trainTime: Double = measures.map(_._3).sum / measures.size.toDouble
    println("Avg time to generate the model: " + trainTime)
    val predTime: Double = measures.map(_._4).sum / measures.size.toDouble
    println("Avg time to predict: " + predTime)


    val inf = new File(args(0))
    val writer = new PrintWriter(new FileWriter(inputFolder+s"res_bag_dbcov_${numModels}_${sampleSize}_${minSupp}.csv", true))
    writer.println(f"${inf.getName}, $accuracy%.4f, $numRules%.4f, $trainTime%.0f, $predTime%.0f, ${accuracies.map(x => f"$x%.4f").mkString(", ")}")
    writer.close()









  }

}
