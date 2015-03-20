import java.io.{File, PrintWriter}

import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Try

/**
 * Created by Luca Venturini on 23/02/15.
 */
object L3uciTests {

  def main(args: Array[String]) {
    val inputFolder = "/home/lucav/data/UCI/test1/"
    val inputFile = "/home/lucav/data/UCI/test1/voting.data" // Should be some file on your system
    val conf = new SparkConf().setAppName("L3Local_UCI_v0.2.0").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val writer = new PrintWriter(new File(inputFolder+"res_nodbcov.csv"))
    val files = new File(inputFolder).listFiles().filter(_.getName.endsWith(".data"))

    for(inputFile <- files) {

      val all = sc.textFile(inputFile.getPath)
      val all2 = {
        if (all.first().startsWith("|")) all.subtract(sc.parallelize(List(all.first()))) else all
      }.filter(_.nonEmpty)
      val data = all2.map(_.split(","))
      val dict = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2, ascending = false).zipWithIndex.collectAsMap
      val transactions = data.map(_.zipWithIndex.map(x => dict(x)))
      val count = transactions.count()
      val numClasses = transactions.map(_.last).max + 1


      val l3 = new L3(numClasses = numClasses.toInt, minSupport = 0.01)






      /* Cross-validation: */
      val numFolds = 10
      val cvTrans = kFold(transactions, numFolds, 12345)
      val measures = cvTrans.map {
        case (train, test) =>
          val t0 = System.nanoTime()
          val model = l3.train(train)
          val t1 = System.nanoTime()
          val labels = test.map(_.find(_ < model.numClasses)) filter (_.nonEmpty) map (_.get)
          val predictions = model.predict(test.map(_.toSet))
          //todo: should we remove the class labels?
          val t2 = System.nanoTime()
          val confusionMatrix = labels.zip(predictions).groupBy(x => x).mapValues(_.size).collectAsMap() //todo:put a countByKey
          (confusionMatrix, model.rules.size, (t1 - t0) / 1e6, (t2 - t1) / 1e6)
      }
      //TODO set params
      val cvConfusionMatrix = measures.map(_._1).reduce(
        //(c1, c2) => c1.map{case(k, v) => (k, v+c2(k))}
        (c1, c2) => (c1.keySet ++ c2.keySet).map(k => (k, c1.getOrElse(k, 0) + c2.getOrElse(k, 0))).toMap
      ).mapValues(_.toDouble / count)



      println(s"Confusion Matrix (avg of $numFolds):")
      val accuracy = cvConfusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum
      println(cvConfusionMatrix.mkString("\n"))
      val numRules: Double = measures.map(_._2).sum / measures.size.toDouble
      println("# of rules: " + numRules)
      println("Accuracy: " + accuracy)
      val trainTime: Double = measures.map(_._3).sum / measures.size.toDouble
      println("Avg time to generate the model: " + trainTime)
      val predTime: Double = measures.map(_._4).sum / measures.size.toDouble
      println("Avg time to predict: " + predTime)


      writer.println(f"${inputFile.getName}, $accuracy%.4f, $numRules%.4f, $trainTime%.0f, $predTime%.0f")

    }
    writer.close()







  }

}
