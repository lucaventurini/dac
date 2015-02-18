/* SimpleApp.scala */

import java.io.{FileWriter, BufferedWriter, File}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.tree.{RandomForest, DecisionTree}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.DenseVector


object BaggingExample {

  def time[A](f: => A, outName: Option[String] = None) = {
    val outStream = outName match{
      case Some(name) => new java.io.FileOutputStream(new java.io.File(name))
      case None => System.out
    }

    val s = System.nanoTime
    val ret = f
    outStream.write({"time: "+(System.nanoTime-s)/1e6+"ms\n"}.getBytes())
    //println("time: "+(System.nanoTime-s)/1e6+"ms")
    ret
  }

  def testMSE(labelsAndPredictions: List[(Double, Double)]) = {
    val vs = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2) }
    val mean = vs.sum / vs.size
    mean
  }

  def main(args: Array[String]) {
    val inputFile = "hdfs://mp1.polito.it/user/lucav/planes/2008.csv" // Should be some file on your system
    val predictionsOutFile = "/home/lucav/predictions.csv"
    val conf = new SparkConf().setAppName("Bagging_v0.1").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val all = sc.textFile(inputFile)

    //val head = all.take(1).head.split(",").toList
    val data = all.filter(!_.startsWith("Year")).map(_.split(","))
    //val features = head.slice(0, 8) ::: head.slice(11, 15) ::: List(head(16), head(17)) ::: head.slice(18, 21)
    //val airports = data.filter(x => x(16)!="NA" && x(17)!="NA").flatMap(x => x(16) :: x(17) :: Nil).distinct.collect().toList.sorted
    val airports = "ABE,ABI,ABQ,ABY,ACK,ACT,ACV,ACY,ADK,ADQ,AEX,AGS,AKN,ALB,ALO,AMA,ANC,ASE,ATL,ATW,AUS,AVL,AVP,AZO,BDL,BET,BFL,BGM,BGR,BHM,BIL,BIS,BJI,BLI,BMI,BNA,BOI,BOS,BPT,BQK,BQN,BRO,BRW,BTM,BTR,BTV,BUF,BUR,BWI,BZN,CAE,CAK,CDC,CDV,CEC,CHA,CHO,CHS,CIC,CID,CLD,CLE,CLL,CLT,CMH,CMI,CMX,COD,COS,CPR,CRP,CRW,CSG,CVG,CWA,CYS,DAB,DAL,DAY,DBQ,DCA,DEN,DFW,DHN,DLG,DLH,DRO,DSM,DTW,EGE,EKO,ELM,ELP,ERI,EUG,EVV,EWN,EWR,EYW,FAI,FAR,FAT,FAY,FCA,FLG,FLL,FLO,FNT,FSD,FSM,FWA,GCC,GEG,GFK,GGG,GJT,GNV,GPT,GRB,GRK,GRR,GSO,GSP,GST,GTF,GTR,GUC,HDN,HHH,HLN,HNL,HOU,HPN,HRL,HSV,HTS,IAD,IAH,ICT,IDA,ILM,IND,INL,IPL,ISP,ITH,ITO,IYK,JAC,JAN,JAX,JFK,JNU,KOA,KTN,LAN,LAS,LAW,LAX,LBB,LCH,LEX,LFT,LGA,LGB,LIH,LIT,LMT,LNK,LRD,LSE,LWB,LWS,LYH,MAF,MBS,MCI,MCN,MCO,MDT,MDW,MEI,MEM,MFE,MFR,MGM,MHT,MIA,MKE,MKG,MLB,MLI,MLU,MOB,MOD,MOT,MQT,MRY,MSN,MSO,MSP,MSY,MTJ,MYR,OAJ,OAK,OGD,OGG,OKC,OMA,OME,ONT,ORD,ORF,OTH,OTZ,OXR,PBI,PDX,PFN,PHF,PHL,PHX,PIA,PIH,PIR,PIT,PLN,PMD,PNS,PSC,PSE,PSG,PSP,PUB,PVD,PWM,RAP,RDD,RDM,RDU,RFD,RHI,RIC,RKS,RNO,ROA,ROC,ROW,RST,RSW,SAN,SAT,SAV,SBA,SBN,SBP,SCC,SCE,SDF,SEA,SFO,SGF,SGU,SHV,SIT,SJC,SJT,SJU,SLC,SLE,SMF,SMX,SNA,SPI,SPS,SRQ,STL,STT,STX,SUN,SUX,SWF,SYR,TEX,TLH,TOL,TPA,TRI,TUL,TUP,TUS,TVC,TWF,TXK,TYR,TYS,VLD,VPS,WRG,WYS,XNA,YAK,YKM,YUM".split(",").toList
    val selected = data .filter(x => x.slice(0,21).forall(_ != "NA"))
      .map { x =>
      //todo → broadcast
      val ori = airports.indexOf(x(16)).toString
      val des = airports.indexOf(x(17)).toString
      val all = x.slice(0, 8) ++ x.slice(11, 15) ++ Array(ori, des) ++ x.slice(18, 21)
      LabeledPoint(x(15).toDouble, new DenseVector(all.map(_.toDouble)))
    }
    // Split the data into training and test sets (30% held out for testing)
    val splits = selected.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    /* RFs */

    // Train a RandomForest model.
    val categoricalFeaturesInfo = Map[Int, Int](12 → airports.size, 13 → airports.size)
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "all"
    val impurity = "variance"
    val maxDepth = 4
    val maxBins = airports.size

    val modelForest = time{RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)}

    // Evaluate model on test instances and compute test error
    val labelsAndPredictionsF = testData.take(100).toList.map { point =>
      val prediction = modelForest.predict(point.features)
      (point.label, prediction)
    }//model.predict(testData.take(1).toList.head.features)

    println("Test Mean Squared Error for RF = " + testMSE(labelsAndPredictionsF))

    /* BAGGING */

    val trees = time{
      Seq.fill(numTrees)(scala.util.Random.nextLong()).
        map(trainingData.sample(true, 0.01, _)).
        map (DecisionTree.trainRegressor(_, categoricalFeaturesInfo,
        impurity,
        maxDepth,
        maxBins))
    }

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.take(100).toList.map { point =>
      val prediction = trees.map(x => x.predict(point.features)) .sum / trees.size
      (point.label, prediction)
    }

    /* OUTPUT */

    println(s"Test Mean Squared Error for Bagging = ${testMSE(labelsAndPredictions)}")

    println("Learned regression forest model:\n" + modelForest.toDebugString)

    println("Learned regression trees models:\n" + trees.map(_.toDebugString))

    // FileWriter
    val file = new File(predictionsOutFile)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(labelsAndPredictionsF.sortBy(_._1).map(_._1).mkString(", ") + "\n")
    bw.write(labelsAndPredictionsF.sortBy(_._1).map(_._2).mkString(", ") + "\n")
    bw.write(labelsAndPredictions.sortBy(_._1).map(_._2).mkString(", ") + "\n")
    bw.close()



  }
}