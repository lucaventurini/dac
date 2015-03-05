import java.io.{File, PrintWriter}

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

    val l3 = new L3(numClasses = 3, minSupport = 0.369) //they start from 1
    val model=l3.train(transactions)
    println(model)

    val writer = new PrintWriter(new File("/home/luca/data/L3.out"))
    writer.write(model.toString())
    writer.close()

    /*val fpg = new FPGrowth()
    	  .setMinSupport(0.2)
    	  .setNumPartitions(10)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach{ itemset => println(itemset._1.mkString("[", ",", "]") + ", " + itemset._2) }

    val numClasses = 2
    model.freqItemsets.map{case (items, sup) => (items.partition(_ < numClasses), sup)}

    val antecedents = model.freqItemsets.map{case (items, sup) => val x = items.partition(_ < numClasses);(x._2.toSet,(x._1, sup))} //cache?
    val supAnts = antecedents.map(x => (x._1, x._2._2)).reduceByKey(_ + _)
    val rules = antecedents.filter(!_._2._1.isEmpty).join(supAnts).mapValues(x => (x._1._1(0), x._1._2, x._1._2/x._2.toFloat))

    rules.collect.foreach(x => println(x._1.mkString(", ") + " -> " + x._2._1 + " (" + x._2._2 + ", " + x._2._3 + ")"))*/
  }

}
