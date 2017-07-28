import java.io.File

import it.polito.dbdmg.ml.DACEnsemble
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
//import play.api.libs.json._


/**
  * Created by lucav on 12/12/16.
  */
object DACUciCVbinary {

  case class Measure(roc_fpr: Array[Double],
                     roc_tpr: Array[Double],
                     auROC: Double,
                     numRules: Int,
                     numRulesII: Int,
                     numRulesLight: Int,
                     lenI: Double,
                     lenII: Double,
                     timeTrain: Double,
                     timePred: Double,
                     methodWeightedVoting: String,
                     metricsValue: String,
                     weight: String,
                     accuracy: Double,
                     curve: Curve
                    )

  case class Curve(accuracy: Array[Double],
                   thresholds: Array[Double])


  def main(args: Array[String]): Unit = {
    if (args.size < 4) return
    val inputFile = args(0)
    val numModels = args(1)
    val sampleSize = args(2)
    val minSupp = args(3)
//    val sampling = ""
    val conf = new SparkConf().setAppName("BAC_UCI_CV_v0.2.0")
    val sc = new SparkContext(conf)

    val all = sc.textFile(inputFile)
    val all2 = {
      if (all.first().startsWith("|")) all.subtract(sc.parallelize(List(all.first()))) else all
    }.filter(_.nonEmpty)
    val data = all2.map(_.split(","))
    val dict = data.map(_.zipWithIndex).flatMap(x => x).distinct.sortBy(_._2, ascending = false).zipWithIndex.collectAsMap
    val trs = data.map(_.zipWithIndex.map(x => dict(x)))
    val count = trs.count()
    val numClasses = trs.map(_.last).max + 1
    val transactions = trs.map {
      x => val (t0, t1) = x.partition(_ < numClasses)
        (t1, t0.head)
    }
    val classCounts = trs.map(_.last).countByValue()
    //    val preferredProba = classCounts.values.map(_.toDouble / count)
    val preferredProba = (0 until numClasses.toInt).map(classCounts(_).toDouble / count)
    println("Class counts: " + classCounts)
    println("default proba: " + preferredProba)


    val metricsValue = "confidence"

    //Grid Validation
//    for (methodWeightedVoting <- Array("mean", "max", "min");
    for (methodWeightedVoting <- Array("max", "min");
//         metricsValue <- Array("confidence", "support");
//         weight <- Array("none", "numRules");
         weight <- Array("none");
         sampling <- Array("oversampling", "undersampling"))
      yield {
        //    val l3 = new L3Ensemble(numModels = numModels.toInt, numClasses = numClasses.toInt, minSupport = minSupp.toDouble, sampleSize = sampleSize.toDouble)
        val l3 = new DACEnsemble(
          strategy = "gain",
          withDBCoverage = false,
          withReplacement = false,
          withShuffling = false,
          saveSpare = false,
          numModels = numModels.toInt,
          classes = (0L until numClasses).toArray,
          numClasses = numClasses.toInt,
          //      preferredClass = Some(0L),
          preferredClass = Some(preferredProba.zipWithIndex.maxBy(_._1)._2.toLong),
          //      preferredProba = Some(Array(1 - ratioClass1, ratioClass1)),
          preferredProba = Some(preferredProba.toArray),
          minSupport = minSupp.toDouble,
          sampleSize = sampleSize.toDouble)

        /* Cross-validation: */
        val numFolds = 10
        val cvTrans = kFold(transactions, numFolds, 12345)

        val collapseMethod = "max"
//        val methodWeightedVoting = "max"
//        val metricsValue = "confidence"
//        val weight = "none"
        println("collapse method: " + collapseMethod)
        println("method weighted voting: " + methodWeightedVoting)
        println("metrics value: " + metricsValue)
        println("weight: " + weight)

        val measures = cvTrans.map {
          case (trainingData, test) =>
//            val ratioOverSample = 138352581 / 4039277765.0 //freq_pos/freq_neg
            val ratioOverSample = classCounts(0)/classCounts(1).toDouble



            val train: RDD[(Array[Long], Long)] = sampling match {
              case "oversampling" => {
                val fractions = Map(0L -> 1.0, 1L -> math.pow(ratioOverSample, -1)) //oversampling 1s
                trainingData.map(x => (x._2, x._1)).sampleByKey(fractions = fractions, withReplacement = true).map(x => (x._2, x._1))

              }
              case "undersampling" => {
                val fractions = Map(0L -> ratioOverSample, 1L -> 1.0) //downsampling 0s
                trainingData.map(x => (x._2, x._1)).sampleByKey(fractions = fractions, withReplacement = true).map(x => (x._2, x._1))
              }

              case x => trainingData
            }

            val t0 = System.nanoTime()
            val model = l3.train(train)
            val t1 = System.nanoTime()

            val numRules: Int = model.totRules
            val numRulesII: Int = model.totRulesII
            val lenI = model.avgItemsInRules
            val lenII = model.avgItemsInRulesII

            val t2 = System.nanoTime()
            val lightModel = model.lightModel(Some(collapseMethod))

            lightModel.setWeightedVoting(methodWeightedVoting)
            lightModel.setMetric(metricsValue)
            lightModel.setWeight(weight)

            val broadcastModel = sc.broadcast(lightModel)

            // Compute raw scores on the test set
            val predictionAndLabels = test.map { x =>
              val prediction = broadcastModel.value.predictProba(x._1.toSeq)(1) //take as score the proba of class 1
            val label = x._2
              (prediction, label.toDouble)
            }

            test.zip(predictionAndLabels).take(5).map(x => x._1._1.mkString(",") + " ==> " + x._2).foreach(println)

            // Instantiate metrics object
            val metrics = new BinaryClassificationMetrics(predictionAndLabels, numBins = 0)

            // ROC Curve
            val roc = metrics.roc
            val roc_fpr = roc.map(_._1).collect()//.sample(false, 0.1, 123).collect()
            val roc_tpr = roc.map(_._2).collect()//.sample(false, 0.1, 123).collect()
            val auROC = metrics.areaUnderROC
            val t3 = System.nanoTime()

            val predAndLabels = predictionAndLabels.map{case (x, y) => (if (x < 0.6) 0.0 else 1.0, y)}
            val metrics2 = new MulticlassMetrics(predAndLabels)

            val cm = metrics2.confusionMatrix
            val accuracy: Double = (cm(0, 0) + cm(1, 1)) / cm.toArray.sum
            println("accuracy: "+accuracy)

            val numRulesLight = lightModel.rules.length
            println("numRules: " + numRules)
            println("numRulesII: " + numRulesII)
            println("lenI " + lenI)
            println("lenII " + lenII)
            println("numRulesLight " + numRulesLight)
            println()
            println("roc_fpr: " + roc_fpr.mkString("[", ",", "]"))
            println("roc_tpr: " + roc_tpr.mkString("[", ",", "]"))
            println("\nauROC: " + auROC)
            println("Time to train (ms): " + (t1 - t0) / 1e6)
            println("Time to test (ms): " + (t3 - t2) / 1e6)
            println("*" * 40)



//            def mapThreshold(newThresholds:Array[Double], realThresholds:Array[Double])={
//              newThresholds.map(x =>
////              realThresholds(realThresholds.lastIndexWhere(_ < x))
////                realThresholds.filter(_ < x).sorted.lastOption.getOrElse(realThresholds.head)
//                realThresholds.filter(_ > x).sorted.headOption.getOrElse(realThresholds.last)
//              )
//            }
//
//            val wantedThresholds = (0.01 to 1.0 by 0.01).toArray
//            val thr1: Array[Double] = metrics.precisionByThreshold().map(_._1).collect()
//            val thrPrec = mapThreshold(wantedThresholds, thr1)
//            val prec = metrics.precisionByThreshold().collectAsMap()
//            println("DEBUG "+prec.keys.mkString(","))
//            val wantedPrec = thrPrec.map(x =>prec(x))
//            println("DEBUG prec: "+wantedPrec.mkString(","))
//
//
//            val thr2: Array[Double] = metrics.recallByThreshold().map(_._1).collect()
//            val thrRecall = mapThreshold(wantedThresholds, thr2)
//
//            val recall = metrics.recallByThreshold().collectAsMap()
//            val wantedRecall = thrRecall.map(x =>recall(x))
//
//            val thr3 = metrics.thresholds().collect()
//            val thrFPR = mapThreshold(wantedThresholds, thr3)
//            val fprByThr = (thr3, roc_fpr).zipped.toMap
//            val wantedFPR = thrFPR.map(x =>fprByThr(x))
//
//
//
//            println("DEBUG thr: "+thr1.mkString(","))
//            println("DEBUG thr: "+thr2.mkString(","))
//            println("DEBUG thr: "+thr3.mkString(","))
//
//            val t = (wantedFPR, wantedPrec, wantedRecall).zipped
//            def accuracyFromCurves(fpr:Double, prec:Double, recall:Double) = {
//              val p1 = prec/(1-prec)
//              val f1 = (1-fpr)/fpr
//              val num = p1 + f1
//              val denum = p1/recall + f1 +1.0
//              num/denum
//            }
//            val accCurve = t.map{case (fpr, prec, recall) => accuracyFromCurves(fpr, prec, recall)
//            }
//
            val wantedThresholds = (0.1 to 1.0 by 0.1).toArray

            val accCurve = wantedThresholds.map { thr =>
              val predAndLabels = predictionAndLabels.map { case (x, y) => (if (x < thr) 0.0 else 1.0, y) }
              val metrics2 = new MulticlassMetrics(predAndLabels)

              val cm = metrics2.confusionMatrix
              val accuracy: Double = (cm(0, 0) + cm(1, 1)) / cm.toArray.sum
              accuracy
            }



            println("DEBUG Accuracy Curve: "+accCurve.mkString(","))
            val curve= new Curve(accCurve,
              wantedThresholds)

            println("DEBUG thresholds: "+curve.thresholds.mkString(","))


            new Measure(roc_fpr,
              roc_tpr,
              auROC,
              numRules,
              numRulesII,
              numRulesLight,
              lenI,
              lenII,
              (t1 - t0) / 1e6,
              (t3 - t2) / 1e6,
              methodWeightedVoting,
              metricsValue,
              weight,
              accuracy = accuracy,
              curve
            )

        }

        def mean(x: Array[Double]): Double = x.sum / x.length
        def variance(x: Array[Double]): Double = {
          val m = mean(x)
          x.map(i => math.pow(i - m, 2)).sum / x.length
        }
        def stddev(x: Array[Double]) = math.sqrt(variance(x))
        def describe(x: Array[Double]) = mean(x) + "  +-" + stddev(x)

        println(measures(1).methodWeightedVoting + " " + measures(1).metricsValue + " " + measures(1).weight)

        println("rules in complete model: " + describe(measures.map(_.numRules.toDouble)))
    val numRulesCV: Array[Double] = measures.map(_.numRulesLight.toDouble)
    println("rules in light model: " + describe(numRulesCV))
        println("# of II level rules in complete model: " + describe(measures.map(_.numRulesII.toDouble)))
        println("avg size of of I level rules: " + describe(measures.map(_.lenI)))
        println("avg size of of II level rules: " + describe(measures.map(_.lenII)))
        println("Time to train (ms): " + describe(measures.map(_.timeTrain)))
        println("Time to test (ms): " + describe(measures.map(_.timePred)))

        println("Area under ROC = " + describe(measures.map(_.auROC)))
        println("Accuracy = " + describe(measures.map(_.accuracy)))

        val accs: Array[Array[Double]] = measures.map(_.curve.accuracy)
        val confs = accs.transpose.map(x => (mean(x), stddev(x)))
    val thr = measures(1).curve.thresholds
    val confCurve = thr.zip(confs)
    val best = confCurve.maxBy(_._2._1)
    println(s"Best Accuracy: ${best._2._1} +- ${best._2._2} at ${best._1}")



        println("FPR:")
        //    println(Json.prettyPrint(Json.toJson(measures.map(_.roc_fpr))))
        println(measures.map(_.roc_fpr.mkString("[", ",", "]")).mkString("[", ",\n", "]"))
        println("TPR:")
        println(measures.map(_.roc_tpr.mkString("[", ",", "]")).mkString("[", ",\n", "]"))

        println("\n")

    val inf = new File(args(0))

    //    println(s"Results of crossvalidation with ${numModels} models, sample size: ${sampleSize}, min supp: ${minSupp}")
    println(f"${inf.getName},$methodWeightedVoting, $metricsValue, $weight, $numClasses,$minSupp, ${best._2._1}%.4f, ${best._2._2}%.4f, ${best._1},  ${mean(numRulesCV)}%.4f, $sampling")


    println("=" * 15)

        //    while(true){}

      }
  }

}