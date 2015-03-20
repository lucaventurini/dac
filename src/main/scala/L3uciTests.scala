import java.io.{File, PrintWriter}

import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Luca Venturini on 23/02/15.
 */
object L3uciTests {

  def main(args: Array[String]) {
    val inputFolder = "/home/lucav/data/UCI/test1/"
    val inputFile = "./src/test/resources/mushroom.dat" // Should be some file on your system
    val conf = new SparkConf().setAppName("L3Local_UCI_v0.2.0").setMaster("local[*]")
    val sc = new SparkContext(conf)


    val all = sc.textFile(inputFile)
    val transactions = all.map(_.split(" ").map(_.toLong))
    val count = transactions.count()

    val l3 = new L3(numClasses = 3, minSupport = 0.2) //they start from 1, minsup=3000:0.369






    /* Cross-validation: */
    val numFolds = 2
    val cvTrans = kFold(transactions, numFolds, 12345)
    val measures = cvTrans.map {
      case (train, test) =>
        val t0 = System.nanoTime()
        val model = l3.train(train)
        val t1 = System.nanoTime()
        //val labels = test.map(_.find(_ < model.numClasses)) filter (_.nonEmpty) map (_.get)
        val predictions = model.predict(test.map(_.toSet))
        //todo: should we remove the class labels?
        val t2 = System.nanoTime()
        val confusionMatrix = predictions.map{case (t, p) => (t.find(_ < model.numClasses).get, p)}.groupBy(x => x).mapValues(_.size).collectAsMap()
        (confusionMatrix, model.rules.count(), (t1-t0)/1e6, (t2-t1)/1e6)
    }
    //TODO set params
    val cvConfusionMatrix = measures.map(_._1).reduce(
          //(c1, c2) => c1.map{case(k, v) => (k, v+c2(k))}
          (c1, c2) => (c1.keySet ++ c2.keySet).map(k => (k, c1.getOrElse(k, 0) + c2.getOrElse(k, 0))).toMap
        ).mapValues(_.toDouble/count)



    println(s"Confusion Matrix (avg of $numFolds):")
    val accuracy = cvConfusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum
    println(cvConfusionMatrix.mkString("\n"))
    val numRules: Double = measures.map(_._2).sum / measures.size.toDouble
    println("# of rules: "+numRules)
    println("Accuracy: "+accuracy)
    val trainTime: Double = measures.map(_._3).sum / measures.size.toDouble
    println("Avg time to generate the model: "+trainTime)
    val predTime: Double = measures.map(_._4).sum / measures.size.toDouble
    println("Avg time to predict: "+predTime)

    val writer = new PrintWriter(new File(inputFolder+"res_nodbcov.csv"))
    writer.println(f"mushrooms, $accuracy%.4f, $numRules%.4f, $trainTime%.0f, $predTime%.0f")
    writer.close()







  }

}
