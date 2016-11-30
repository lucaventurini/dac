/**
  * Created by ernesto on 29/11/16.
  */
import java.io.{BufferedWriter, File, FileWriter, OutputStreamWriter}

import it.polito.dbdmg.ml.L3Ensemble
import it.polito.dbdmg.ml.Rule
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem => HDFS, Path}
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.DoubleRDDFunctions
import org.apache.spark.{SparkConf, SparkContext}
import org.github.jamm.MemoryMeter


object BacNetCv {

  def main(args: Array[String]) {

    if (args.size < 5) return
    val id = args(0).toInt
    val inputFile = args(1)
    val numModels = args(2).toInt
    val sampleSize = args(3).toDouble
    val minSupp = args(4).toDouble
    try {
      val conf = new SparkConf()
      val sc = new SparkContext(conf)

      val all = sc.textFile(inputFile)
      val header = all.first().split(" ").zipWithIndex.map(t => (t._2,t._1)).toMap
      val all2 = {
        if (all.first().startsWith("#")) all.subtract(sc.parallelize(List(all.first()))) else all
      }.filter(_.nonEmpty)
      val all3 = all2.map(_.split(" "))

      /*//takes only the first 20 columns + class label
      val selectedCols = all3.map(_.zipWithIndex).map {
        x => val f = x.filter(v => v._2 < 20 || v._2 == x.length-1)
          f.map(_._1)
      }

      val data = selectedCols
  */
      val data = all3
      val numPartitions = data.partitions.size
      /*val indexes = data.zipWithIndex().map(x => (x._1.zipWithIndex.map(k => ((k._1,k._2),x._2)))).flatMap(x => x)
      val values = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2, ascending = false).zipWithIndex
      val transactions = indexes.join(values).groupBy(x => x._2._1).map(x => x._2.toArray.sortBy(_._1._2).map(_._2._2))
      val id2val = values.map(_.swap)*/

      val dict = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2, ascending = false).zipWithIndex.collectAsMap()
      val id2val = dict.map(_.swap)
      val trs = data.map(_.zipWithIndex.map(x => dict(x)))
      val numClasses = trs.map(_.last).max + 1
      val transactions = trs.map {
        x => val (t0, t1) = x.partition(_ < numClasses)
          (t1, t0.head)
      }

      //val count = transactions.count()

      val l3 = new L3Ensemble(numModels = numModels.toInt, numClasses = numClasses.toInt, minSupport = minSupp.toDouble, sampleSize = sampleSize.toDouble)

      val t0 = System.nanoTime()
      val model = l3.train(transactions)
      val t1 = System.nanoTime()
      val trainTime = (t1 - t0) / 1e6
      //val rules_file = new File("/home/ernesto/Documenti/video_log/rules.txt")
      //val bw = new BufferedWriter(new FileWriter(rules_file))

      println("Tot rules: " + model.totRules)
      println("Tot rules II: " + model.totRulesII)
      println("Avg items in rules: " + model.avgItemsInRules)
      println("Avg items in rulesII: " + model.avgItemsInRulesII)
      println("Training time: " + trainTime)
      println("Classes: "+numClasses)
      println()



      /*//write on normal file system
      bw.write("Tot rules:\t\t" + model.totRules)
      bw.write("\nTot rules II:\t\t" + model.totRulesII)
      bw.write("\nAvg items in rules:\t\t" + model.avgItemsInRules)
      bw.write("\nAvg items in rulesII: " + model.avgItemsInRulesII)
      bw.write("\nTraining time:\t\t" + trainTime)
      bw.write("\n")

      model.models.foreach {
        m => m.rules.foreach {
          r => val ants = r.antecedent.map(x => id2val(x))
            val cons = id2val(r.consequent)._1
            val headAnts = ants.map(x => "" + header(x._2) + "=" + x._1)
            val str = headAnts.mkString(" ") + f" -> $cons (sup=${r.support}%.6f, conf=${r.confidence}%.6f, chi2=${r.chi2}%.6f)"
            println(str)
            bw.write(str + "\n")
        }
      }*/

      //write to hdfs
      val hdfsConf = new Configuration()
      hdfsConf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
      hdfsConf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
      val hdfs = HDFS.newInstance(hdfsConf)
      val path = new Path(f"/user/valentino/rules_2_$minSupp.txt")
      if (hdfs.exists(path))
        hdfs.delete(path, true)
      val os = hdfs.create(path)
      val bw = new BufferedWriter(new OutputStreamWriter(os))

      bw.write("Tot rules:\t\t" + model.totRules)
      //bw.write("\nTot rules II:\t\t" + model.totRulesII)
      bw.write("\nAverage items in rules:\t\t" + model.avgItemsInRules)
      //bw.write("\nAvg items in rulesII: " + model.avgItemsInRulesII)
      bw.write("\nTraining time(ms):\t\t" + trainTime)
      bw.write("\n")

      model.models.foreach {
        m => m.rules.foreach {
          r => val ants = r.antecedent.map(x => id2val(x))
            val cons = id2val(r.consequent)._1
            val headAnts = ants.map(x => "" + header(x._2) + "=" + x._1)
            val str = headAnts.mkString(" ") + f" -> $cons (sup=${r.support}%.6f, conf=${r.confidence}%.6f, chi2=${r.chi2}%.6f)"
            println(str)
            bw.write(str + "\n")
        }
      }

      bw.close()

      val path2 = new Path(f"/user/valentino/best_rules_2_$minSupp.txt")
      if (hdfs.exists(path2))
        hdfs.delete(path2, true)
      val os2 = hdfs.create(path2)
      val bw2 = new BufferedWriter(new OutputStreamWriter(os2))

      //      val br_file = new File("/home/ernesto/Documenti/video_log/best_rules.txt")
      //      val bw2 = new BufferedWriter(new FileWriter(rules_file))

      val bestRules = model.models.map(_.rules.sortBy(_.support).reverse.take(2))

      /*bestRules.foreach{
        x => x.foreach{
          r => val ants = r.antecedent.map(x => id2val.lookup(x).head)
            val cons = id2val.lookup(r.consequent).head._1
            val headAnts = ants.map(x => "" + header(x._2) + "=" + x._1)
            val str = headAnts.mkString(" ") + f" -> $cons (sup=${r.support}%.6f, conf=${r.confidence}%.6f, chi2=${r.chi2}%.6f)"
            bw2.write(str + "\n")
        }
      }*/

      bestRules.foreach{
        x => x.foreach{
          r => val ants = r.antecedent.map(x => id2val(x))
            val cons = id2val(r.consequent)._1
            val headAnts = ants.map(x => "" + header(x._2) + "=" + x._1)
            val str = headAnts.mkString(" ") + f" -> $cons (sup=${r.support}%.6f, conf=${r.confidence}%.6f, chi2=${r.chi2}%.6f)"
            bw2.write(str + "\n")
        }
      }

      bw2.close()
      hdfs.close()





      /* Cross-validation: */

      cv(transactions = transactions, l3 = l3)

      def cv(transactions: RDD[(Array[Long], Long)], l3: L3Ensemble): Unit = {
        val count = transactions.count()
        val numClasses = transactions.map(_._2).max + 1
        val numFolds = 10
        val cvTrans = kFold(transactions, numFolds, 12345)
        val measures = cvTrans.map {
          case (train, test) =>
            val t0 = System.nanoTime()
            val model = l3.train(train)
            val t1 = System.nanoTime()
            val labels = test.map(_._2)
            val t2 = System.nanoTime()
            val predictions = model.predict(test.map(_._1.toSet))
            val t3 = System.nanoTime()
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
              (t3 - t2) / 1e6,
              ((t3 - t2) / 1e6) / test.count(),
              accuracy)
        }
        //TODO set params
        val cvConfusionMatrix = measures.map(_._1).reduce(
          //(c1, c2) => c1.map{case(k, v) => (k, v+c2(k))}
          (c1, c2) => (c1.keySet ++ c2.keySet).map(k => (k, c1.getOrElse(k, 0) + c2.getOrElse(k, 0))).toMap
        ).mapValues(_.toDouble / count).map{
          x => ((id2val(x._1._1)._1,id2val(x._1._2)._1),x._2)
        }.toMap



        /*val out = new File("/home/ernesto/Documenti/video_log/cv.txt")
        val cm_out = new File("/home/ernesto/Documenti/video_log/cv.csv")
        val bw = new BufferedWriter(new FileWriter(out))
        val bw2 = new BufferedWriter(new FileWriter(cm_out))
        cvConfusionMatrix.foreach(x => bw2.write(x._1._1+","+x._1._2+","+x._2+"\n"))
        bw2.close()*/

        val hdfsConf = new Configuration()
        hdfsConf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
        hdfsConf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
        val hdfs = HDFS.newInstance(hdfsConf)
        val path = new Path(f"/user/valentino/cv_2_$minSupp.csv")
        if (hdfs.exists(path))
          hdfs.delete(path, true)
        val os = hdfs.create(path)
        val bw2 = new BufferedWriter(new OutputStreamWriter(os))
        cvConfusionMatrix.foreach(x => bw2.write(x._1._1+","+x._1._2+","+x._2+"\n"))
        bw2.close()

        val path2 = new Path(f"/user/valentino/cv_2_$minSupp.txt")
        if (hdfs.exists(path2))
          hdfs.delete(path2, true)
        val os2 = hdfs.create(path2)
        val bw = new BufferedWriter(new OutputStreamWriter(os2))

        println(s"Confusion Matrix for $inputFile (avg of $numFolds):")
        val accuracies = measures.map(_._8)
        //val accuracy = cvConfusionMatrix.filterKeys(x => x._1 == x._2).map(_._2).sum
        val accuracy = accuracies.sum / numFolds
        val variance = (accuracies.map(x => math.pow(x - accuracy, 2)).sum -
          math.pow(accuracies.map(_ - accuracy).sum, 2)/numFolds) / numFolds
        println(cvConfusionMatrix.mkString("\n"))

        bw.write(s"Confusion Matrix for $inputFile (avg of $numFolds):\n")
        bw.write(cvConfusionMatrix.mkString("\n"))

        val numRules: Double = measures.map(_._2).sum / measures.size.toDouble
        val numRulesII: Double = measures.map(_._3).sum / measures.size.toDouble
        val numItemsInRules: Double = measures.map(_._4).sum / measures.size.toDouble
        println("# of rules: " + numRules)
        println("# of II level rules: " + numRulesII)
        println("# of items in all the rules (antecedents): " + numItemsInRules)
        println(s"Accuracy: $accuracy, variance: $variance")

        bw.write("\n# of rules:\t" + numRules)
        bw.write("\n# of II level rules:\t" + numRulesII)
        bw.write("\n# of items in all the rules (antecedents):\t" + numItemsInRules)
        bw.write(s"\nAccuracy:\t$accuracy\nVariance:\t$variance")

        val trainTime: Double = measures.map(_._5).sum / measures.size.toDouble
        println("Avg time to generate the model: " + trainTime)
        bw.write("\nAvg time to generate the model:\t" + trainTime)
        val predTime: Double = measures.map(_._6).sum / measures.size.toDouble
        val pred1: Double = measures.map(_._7).sum / measures.size.toDouble
        println("Avg time to predict: " + predTime)
        bw.write("\nAvg time to predict:\t" + predTime)
        println("Avg time to predict one record: " + pred1)
        bw.write("\nAvg time to predict one record:\t" + pred1)
        bw.close()

        val out = (f"${id}%4d;${numModels.toInt}%15d;${sampleSize}%20.6f;${minSupp}%20.6f;${numClasses}%15d;" +
          f"${numRules}%10.6f;${numRulesII}%10.6f;${accuracy}%10.8f;${variance}%20.9f;${trainTime}%10.6f;" +
          f"${predTime}%10.8f;${pred1}%10.10f\n")
        val path3 = new Path(f"/user/valentino/tmp_${id}%03d")
        if (hdfs.exists(path3))
          hdfs.delete(path3, true)

        val os3 = hdfs.create(path3)
        val bw3 = new BufferedWriter(new OutputStreamWriter(os3))

        bw3.write(out+"\n")
        bw3.close()


        bw.close()

        hdfs.close()
      }
    }
    catch {
      case ex: Exception => {
        System.err.println(ex.getMessage)
        System.err.println(ex.getCause)
        System.err.println(ex.printStackTrace())
      }
    }
  }
}
