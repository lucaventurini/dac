import java.util.Locale
import java.io._

import it.polito.dbdmg.ml.L3Ensemble

import org.apache.spark.{SparkConf, SparkContext, SparkException}

import org.github.jamm.MemoryMeter

import org.apache.hadoop.fs.{Path, FileSystem => HDFS}
import org.apache.hadoop.conf._

/**
  * Created by ernesto on 26/05/16.
  */
object Bac2Experimental {

  def main(args: Array[String]): Unit = {

    if (args.size < 6) return
    val id = args(0).toInt
    val numModels = args(1).toInt
    val minSupp = args(2).toDouble
    val sampleSize = args(3).toDouble
    val numBins = args(4).toInt
    val inputFile = args(5)
    var tuples = 0L
    var numAtt = 0
    var numClasses = 0L

    Locale.setDefault(new Locale("en", "US"))


    try
    {
      val conf = new SparkConf()
      val sc = new SparkContext(conf)

      val all = sc.textFile(inputFile)
      if (all.isEmpty()) {
        System.err.println("Empty collection")
        return
      }

      tuples = all.count()
      val data = all.map(_.split(";"))
      val dict = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2, ascending = false).zipWithIndex.collectAsMap
      val trs = data.map(_.zipWithIndex.map(x => dict(x)))
      numClasses = trs.map(_.last).max + 1
      val transactions = trs.map {
        x => val (t0, t1) = x.partition(_ < numClasses)
          (t1, t0.head)
      }
      val count = transactions.count()
      numAtt = data.first().length - 1

      val l3 = new L3Ensemble(numModels = numModels,
        numClasses = numClasses.toInt,
        minSupport = minSupp,
        sampleSize = sampleSize,
        strategy = "gain")
      val t0 = System.nanoTime()
      val model = l3.train(transactions)
      val t1 = System.nanoTime()
      val trainTime = (t1 - t0) / 1e6

      val meter = new MemoryMeter
      val bytesRules = model.models.map {
        m => m.rules.map {
          r => (meter.measure(r) +
            meter.measure(r.antecedent) +
            meter.measure(r.consequent) +
            meter.measure(r.chi2) +
            meter.measure(r.confidence) +
            meter.measure(r.support))
        }.sum
      }.sum / 1024.0

      val bytesRulesII = model.models.map {
        m => m.rulesIIlevel.map {
          r => (meter.measure(r) +
            meter.measure(r.antecedent) +
            meter.measure(r.consequent) +
            meter.measure(r.chi2) +
            meter.measure(r.confidence) +
            meter.measure(r.support))
        }.sum
      }.sum / 1024.0

      val out = (f"${id}%4d;${numModels.toInt}%15d;${sampleSize}%20.6f;${minSupp}%20.6f;${numClasses}%15d;" +
        f"$tuples%15d;$numAtt%15d;$numBins%15d;${model.totRules}%15d;${model.totRulesII}%15d;${model.avgItemsInRules}%20.6f; " +
        f"${model.avgItemsInRulesII}%20.6f;${bytesRules}%20.2f;${bytesRulesII}%21.2f;${trainTime}%20.6f\n")

      println(out)

      val hdfsConf = new Configuration()
      hdfsConf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
      hdfsConf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
      val hdfs = HDFS.newInstance(hdfsConf)
      val path = new Path(f"/user/valentino/tmp_${id}%04d")
      if (hdfs.exists(path))
        hdfs.delete(path, true)
      val os = hdfs.create(path)
      val bw = new BufferedWriter(new OutputStreamWriter(os))

      bw.write(out)
      bw.close()

      hdfs.close()

    } catch {
      case se: SparkException => {
        System.err.println(se.getMessage)
        System.err.println(se.getCause)
        System.err.println(se.printStackTrace())

        val out = (f"${id}%4d;${numModels.toInt}%15d;${sampleSize}%20.6f;${minSupp}%20.6f;${numClasses}%15d;" +
          f"${tuples}%15d;${numAtt}%15d;$numBins%15d;${0}%15d;${0}%15d;${0.0}%20.6f;${0.0}%20.6f;${0.0}%20.2f;${0.0}%21.2f;${0.0}%20.6f\n")

        val hdfsConf = new Configuration()
        hdfsConf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
        hdfsConf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
        val hdfs = HDFS.newInstance(hdfsConf)
        val path = new Path(f"/user/valentino/tmp_${id}%04d")
        if (hdfs.exists(path))
          hdfs.delete(path, true)

        val os = hdfs.create(path)
        val bw = new BufferedWriter(new OutputStreamWriter(os))

        bw.write(out)
        bw.close()

        hdfs.close()
        throw se

      }
      case io: IOException => {
        System.err.println(io.getMessage)
        System.err.println(io.getCause)
        System.err.println(io.printStackTrace())
        throw io
      }
      case ex: Exception => {
        System.err.println(ex.getMessage)
        System.err.println(ex.getCause)
        System.err.println(ex.printStackTrace())

        val out = (f"${id}%4d;${numModels.toInt}%15d;${sampleSize}%20.6f;${minSupp}%20.6f;${numClasses}%15d;" +
          f"${tuples}%15d;${numAtt}%15d;$numBins%15d;${0}%15d;${0}%15d;${0.0}%20.6f;${0.0}%20.6f;${0.0}%20.2f;${0.0}%21.2f;${0.0}%20.6f\n")

        val hdfsConf = new Configuration()
        hdfsConf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
        hdfsConf.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
        val hdfs = HDFS.newInstance(hdfsConf)
        val path = new Path(f"/user/valentino/tmp_${id}%04d")
        if (hdfs.exists(path))
          hdfs.delete(path, true)

        val os = hdfs.create(path)
        val bw = new BufferedWriter(new OutputStreamWriter(os))

        bw.write(out)
        bw.close()

        hdfs.close()
        throw ex
      }
    }
  }

}
