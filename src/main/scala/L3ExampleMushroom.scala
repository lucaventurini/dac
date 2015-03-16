import java.io.{File, PrintWriter}

import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by Luca Venturini on 23/02/15.
 */
object L3ExampleMushroom {

  def main(args: Array[String]) {
    val inputFile = "/home/luca/data/mushroom/mushroom.dat" // Should be some file on your system
    val conf = new SparkConf().setAppName("Bagging_v0.2.0").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val all = sc.textFile(inputFile)

    val transactions = all.map(_.split(" ").map(_.toLong))
    val count = transactions.count()

    /* On all the dataset: */
    val l3 = new L3(numClasses = 3, minSupport = 0.2, minChi2 = 0) //they start from 1, minsup=3000:0.369
    val model=l3.train(transactions)
    println(model)

    var writer = new PrintWriter(new File("/home/luca/data/L3.out"))
    writer.write(model.toString())
    writer.close()



    /* Cross-validation: */
    val numFolds = 4
    val cvTrans = kFold(transactions, numFolds, 12345)
    val measures = cvTrans.map {
      case (train, test) =>
        val t0 = System.nanoTime()
        val model = l3.train(train).dBCoverage()
        val t1 = System.nanoTime()
        val labels = test.map(_.find(_ < model.numClasses)) filter (_.nonEmpty) map (_.get)
        val predictions = model.predict(test.map(_.toSet)) //todo: should we remove the class labels?
      val t2 = System.nanoTime()
      val confusionMatrix = labels.zip(predictions).groupBy(x => x).mapValues(_.size).collectAsMap()
        (confusionMatrix, model.rules.size, (t1-t0)/1e6, (t2-t1)/1e6)
    }
    //TODO set params
    val cvConfusionMatrix = measures.map(_._1).reduce(
          //(c1, c2) => c1.map{case(k, v) => (k, v+c2(k))}
          (c1, c2) => (c1.keySet ++ c2.keySet).map(k => (k, c1.getOrElse(k, 0) + c2.getOrElse(k, 0))).toMap
        ).mapValues(_.toDouble/count)



    println(s"Confusion Matrix (avg of $numFolds):")
    println(cvConfusionMatrix.mkString("\n"))
    println("# of rules: "+measures.map(_._2).sum/measures.size.toDouble)
    println("Avg time to generate the model: "+measures.map(_._3).sum/measures.size.toDouble)
    println("Avg time to predict: "+measures.map(_._4).sum/measures.size.toDouble)





  }

}
