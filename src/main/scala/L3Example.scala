import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by Luca Venturini on 23/02/15.
 */
object L3Example {

  def main(args: Array[String]) {
    val inputFile = "/home/luca/data/voting/house-votes-84.data" // Should be some file on your system
    val predictionsOutFilePrefix = "predictions"
    val conf = new SparkConf().setAppName("Bagging_v0.2.0").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val all = sc.textFile(inputFile)

    val data = all.map(_.split(","))
    val dict = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2).zipWithIndex.collectAsMap
    val transactions = data.map(_.zipWithIndex.map(x => dict(x)))

    val fpg = new FPGrowth()
    	  .setMinSupport(0.2)
    	  .setNumPartitions(10)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach{ itemset => println(itemset._1.mkString("[", ",", "]") + ", " + itemset._2) }

  }

}
