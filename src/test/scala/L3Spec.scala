import java.util.Locale

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest._
import it.polito.dbdmg.ml._

import scala.util.Random

/**
  * Created by luca on 25/02/15.
  */

trait MLlibTestSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("MLlibUnitTest")
    sc = new SparkContext(conf)
    Locale.setDefault(new Locale("en", "US"))
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    super.afterAll()
  }
}


class L3Spec extends FlatSpec with ShouldMatchers with MLlibTestSparkContext{

  val input = List[Array[Long]](
    Array(0, 10, 11, 12),
    Array(1, 20, 21, 22),
    Array(0, 20, 21, 22),
    Array(0, 20, 21, 22, 23),
    Array(0, 20, 21),
    Array(1, 21, 22)
  )
  val data = input.map {
    x => val (p0, p1) = x.partition(_ < 2)
      (p1 ,p0.head)
  }
  lazy val l3 = new L3(numClasses = 2, minChi2 = 0.0, strategy = "support")
  lazy val model = l3.train(data)
  lazy val modelCovered = model.dBCoverage(data)


    // Version 1

    "The L3 rule extractor" should "extract rules" in {
      new L3(numClasses = 2, minChi2 = 0.0, strategy = "support").train(List[(Array[Long], Long)](
        (Array(10, 11, 12), 0),
        (Array(11, 12), 0),
        (Array(12), 1)
      )).toString().split("\n") should equal(
        """11 12 -> 0 (0.666667, 1.000000, 3.000000)
          |11 -> 0 (0.666667, 1.000000, 3.000000)
          |10 11 12 -> 0 (0.333333, 1.000000, 0.750000)
          |10 12 -> 0 (0.333333, 1.000000, 0.750000)
          |10 11 -> 0 (0.333333, 1.000000, 0.750000)
          |10 -> 0 (0.333333, 1.000000, 0.750000)
          |12 -> 0 (0.666667, 0.666667, NaN)""".stripMargin.split("\n"))
      /* without filters:
       """10 12 -> 0 (0.333333, 1.000000, 0.750000)
          |10 11 -> 0 (0.333333, 1.000000, 0.750000)
          |11 -> 0 (0.666667, 1.000000, 3.000000)
          |10 -> 0 (0.333333, 1.000000, 0.750000)
          |11 12 -> 0 (0.666667, 1.000000, 3.000000)
          |10 11 12 -> 0 (0.333333, 1.000000, 0.750000)
          |12 -> 0 (0.666667, 0.666667, NaN)
          |12 -> 1 (0.333333, 0.333333, NaN)"""
       */


    }

    it should "extract rules and filter confidence" in {
      model.toString().split("\n") should equal(
        """20 21 -> 0 (0.500000, 0.750000, 0.375000)
          |20 -> 0 (0.500000, 0.750000, 0.375000)
          |22 20 21 -> 0 (0.333333, 0.666667, 0.000000)
          |22 20 -> 0 (0.333333, 0.666667, 0.000000)
          |21 -> 0 (0.500000, 0.600000, 0.600000)
          |22 21 -> 1 (0.333333, 0.500000, 1.500000)
          |22 21 -> 0 (0.333333, 0.500000, 1.500000)
          |22 -> 1 (0.333333, 0.500000, 1.500000)
          |22 -> 0 (0.333333, 0.500000, 1.500000)""".stripMargin.split("\n"))

      /* without filters:
      """22 -> 0 (0.333333, 0.500000, 1.500000)
        |22 -> 1 (0.333333, 0.500000, 1.500000)
        |22 21 -> 0 (0.333333, 0.500000, 1.500000)
        |22 21 -> 1 (0.333333, 0.500000, 1.500000)
        |20 21 -> 0 (0.500000, 0.750000, 0.375000)
        |21 -> 0 (0.500000, 0.600000, 0.600000)
        |21 -> 1 (0.333333, 0.400000, 0.600000)
        |22 20 -> 0 (0.333333, 0.666667, 0.000000)
        |22 20 21 -> 0 (0.333333, 0.666667, 0.000000)
        |20 -> 0 (0.500000, 0.750000, 0.375000)"""
        */
    }




  /*ignore should "filter chi2" in {
    fail()
  }*/

  "The L3 model" should "predict a single value" in {
    model.predict(Set(20L, 22L)) should equal(0)
  }

  it should "predict when we have a superset of the items" in {
    model.predict(Set(20L, 45L, 22L)) should equal(0)
  }

  it should "predict something for items never seen" in {
    model.predict(Set(30L)) should equal(0)
  }

  it should "predict an RDD of values" in {
    model.predict(sc.parallelize(List(
      Set(20L, 22L),
      Set(21L)
    ))).count() should equal(2)
  }

  "The DB coverage phase" should "filter harmful rules" in {
    val l: List[(Array[Long], Long)] = List[(Array[Long], Long)]((Array(20L, 21L, 22L), 1L))
    model.dBCoverage(l).toString.split("\n") should
      equal("""22 21 -> 1 (0.333333, 0.500000, 1.500000)
              |22 21 -> 0 (0.333333, 0.500000, 1.500000)
              |22 -> 1 (0.333333, 0.500000, 1.500000)
              |22 -> 0 (0.333333, 0.500000, 1.500000)""".stripMargin.split("\n"))
  }

  it should "do something" in {
    //print the model
    modelCovered.toString.split("\n") should equal ("""20 21 -> 0 (0.500000, 0.750000, 0.375000)
                                                      |22 21 -> 1 (0.333333, 0.500000, 1.500000)
                                                      |20 -> 0 (0.500000, 0.750000, 0.375000)
                                                      |22 20 21 -> 0 (0.333333, 0.666667, 0.000000)
                                                      |22 20 -> 0 (0.333333, 0.666667, 0.000000)
                                                      |22 21 -> 0 (0.333333, 0.500000, 1.500000)
                                                      |22 -> 1 (0.333333, 0.500000, 1.500000)
                                                      |22 -> 0 (0.333333, 0.500000, 1.500000)""".stripMargin.split("\n"))
  }

  val s = Seq.fill(500)(Array(math.round(Random.nextFloat()).toLong, 2+(10*Random.nextDouble()).round, 13, 14))
  val labeledPoints = s.map {
    x => val (t0, t1) = x.partition(_ < 2)
      (t1, t0.head)
  }
  lazy val modelbag = new L3Ensemble(numClasses = 2, numModels = 2, strategy = "support").train(sc.parallelize(labeledPoints))

  "Bagging" should "do something" in {
    modelbag.predict(Set(10L,12L)) should (equal(0) or equal(1))
  }

  it should "extract some rules" in {
    modelbag.toString().split("\n").size should be >= 4
  }


  "On Mushroom" should "extract 137 rules, with sup=3000 and conf=0.5" in {
    val inputFile = "./src/test/resources/mushroom.dat"
    val all = sc.textFile(inputFile)
    val transactions = all.map(_.split(" ").map(_.toLong)).collect().map {
      x => val (t0, t1) = x.partition(_ < 3)
        (t1, t0.head)
    }

    val l3 = new L3(numClasses = 3, minSupport = 0.369, minChi2 = 0, strategy = "support") //they start from 1, minsup=3000

    val model=l3.train(transactions)

    model.rules should have size 137
  }

}

class L3LocalSpec extends FlatSpec with ShouldMatchers with MLlibTestSparkContext{

  val input = List[(Array[Long], Long)](
    (Array(10, 11, 12),0),
    (Array(20, 21, 22),1),
    (Array(20, 21, 22),0),
    (Array(20, 21, 22, 23),0),
    (Array(20, 21),0),
    (Array(21, 22),1)
  )
  lazy val model:L3LocalModel = {new L3(numClasses = 2, minChi2 = 0.0, strategy = "support").train(input)}

  lazy val modelCovered = model.dBCoverage(input)

  "The L3 Local rule extractor" should "extract rules" in {
    new L3(numClasses = 2, minChi2 = 0.0, strategy = "support").train(List[(Array[Long], Long)](
      (Array(10, 11, 12),0),
      (Array(11, 12),0),
      (Array(12),1)
    )).toString().split("\n") should equal(
      """11 12 -> 0 (0.666667, 1.000000, 3.000000)
        |11 -> 0 (0.666667, 1.000000, 3.000000)
        |10 11 12 -> 0 (0.333333, 1.000000, 0.750000)
        |10 12 -> 0 (0.333333, 1.000000, 0.750000)
        |10 11 -> 0 (0.333333, 1.000000, 0.750000)
        |10 -> 0 (0.333333, 1.000000, 0.750000)
        |12 -> 0 (0.666667, 0.666667, NaN)""".stripMargin.split("\n"))
    /* without filters:
     """10 12 -> 0 (0.333333, 1.000000, 0.750000)
        |10 11 -> 0 (0.333333, 1.000000, 0.750000)
        |11 -> 0 (0.666667, 1.000000, 3.000000)
        |10 -> 0 (0.333333, 1.000000, 0.750000)
        |11 12 -> 0 (0.666667, 1.000000, 3.000000)
        |10 11 12 -> 0 (0.333333, 1.000000, 0.750000)
        |12 -> 0 (0.666667, 0.666667, NaN)
        |12 -> 1 (0.333333, 0.333333, NaN)"""
     */


  }

  it should "extract rules and filter confidence" in {
    model.toString().split("\n") should equal(
      """20 21 -> 0 (0.500000, 0.750000, 0.375000)
        |20 -> 0 (0.500000, 0.750000, 0.375000)
        |22 20 21 -> 0 (0.333333, 0.666667, 0.000000)
        |22 20 -> 0 (0.333333, 0.666667, 0.000000)
        |21 -> 0 (0.500000, 0.600000, 0.600000)
        |22 21 -> 1 (0.333333, 0.500000, 1.500000)
        |22 21 -> 0 (0.333333, 0.500000, 1.500000)
        |22 -> 1 (0.333333, 0.500000, 1.500000)
        |22 -> 0 (0.333333, 0.500000, 1.500000)""".stripMargin.split("\n"))

    /* without filters:
    """22 -> 0 (0.333333, 0.500000, 1.500000)
      |22 -> 1 (0.333333, 0.500000, 1.500000)
      |22 21 -> 0 (0.333333, 0.500000, 1.500000)
      |22 21 -> 1 (0.333333, 0.500000, 1.500000)
      |20 21 -> 0 (0.500000, 0.750000, 0.375000)
      |21 -> 0 (0.500000, 0.600000, 0.600000)
      |21 -> 1 (0.333333, 0.400000, 0.600000)
      |22 20 -> 0 (0.333333, 0.666667, 0.000000)
      |22 20 21 -> 0 (0.333333, 0.666667, 0.000000)
      |20 -> 0 (0.500000, 0.750000, 0.375000)"""
      */
  }



  /*ignore should "filter chi2" in {
    fail()
  }*/

  "The L3 Local model" should "predict a single value" in {
    model.predict(Set(20L, 22L)) should equal(0)
    model.predict(Seq(20L, 22L)) should equal(0)
  }

  it should "predict when we have a superset of the items" in {
    model.predict(Set(20L, 45L, 22L)) should equal(0)
    model.predict(Seq(20L, 45L, 22L)) should equal(0)
  }

  it should "predict something for items never seen" in {
    model.predict(Set(30L)) should equal(0)
    model.predict(Seq(30L)) should equal(0)
  }

  it should "predict an RDD of values" in {
    model.predict(sc.parallelize(List(
      Set(20L, 22L),
      Set(21L)
    ))).count() should equal(2)
  }

  it should "predict probabilities" in {
    model.predictProba(Seq(20L, 22L)).sum should equal(1.0)
    model.predictProba(Seq(20L, 45L, 22L)).sum should equal(1.0)
    model.predictProba(Seq(30L)).sum should equal(1.0)
    model.classes should have size(2L)
    model.predictProba(Seq(30L)) should have size(2L)
    modelCovered.predictProba(Seq(20L)).sum should equal(1.0)
  }

  it should "merge" in {
    val oth = model.merge(model)
    oth.rules.size should equal(model.rules.size)
    oth.rulesIIlevel.size should equal(model.rulesIIlevel.size)
  }

  "The DB coverage phase" should "filter harmful rules" in {
    model.dBCoverage(List((Array[Long](20, 21, 22),1L))).toString.split("\n") should
      equal("""22 21 -> 1 (0.333333, 0.500000, 1.500000)
              |22 21 -> 0 (0.333333, 0.500000, 1.500000)
              |22 -> 1 (0.333333, 0.500000, 1.500000)
              |22 -> 0 (0.333333, 0.500000, 1.500000)""".stripMargin.split("\n"))
  }

  it should "do something" in {
    //print the model
    modelCovered.toString.split("\n") should equal ("""20 21 -> 0 (0.500000, 0.750000, 0.375000)
                                                      |22 21 -> 1 (0.333333, 0.500000, 1.500000)
                                                      |20 -> 0 (0.500000, 0.750000, 0.375000)
                                                      |22 20 21 -> 0 (0.333333, 0.666667, 0.000000)
                                                      |22 20 -> 0 (0.333333, 0.666667, 0.000000)
                                                      |22 21 -> 0 (0.333333, 0.500000, 1.500000)
                                                      |22 -> 1 (0.333333, 0.500000, 1.500000)
                                                      |22 -> 0 (0.333333, 0.500000, 1.500000)""".stripMargin.split("\n"))
  }

  it should "not save spare rules if asked so" in {
    model.dBCoverage(List((Array[Long](20, 21, 22), 1L)), saveSpare = false).rulesIIlevel should have size 0
  }


  val s = Seq.fill(500)(Array(math.round(Random.nextFloat()).toLong, 2+(10*Random.nextDouble()).round, 13, 14))
  val labeledPoints = s.map {
    x => val (t0, t1) = x.partition(_ < 2)
      (t1, t0.head)
  }
  lazy val modelbag = new L3Ensemble(numClasses = 2, numModels = 2, strategy = "support").train(sc.parallelize(labeledPoints))

  "Bagging" should "do something" in {
    modelbag.predict(Set(10L,12L)) should (equal(0) or equal(1))
  }

  it should "extract some rules" in {
    modelbag.toString().split("\n").size should be >= 4
  }

  it should "predict an RDD" in {
    modelbag.predict(sc.parallelize(List(Set(10L,12L)))).count() should equal(1)
    modelbag.predict(sc.parallelize(List(Set(10L,12L)))).first() should (equal(0) or equal(1))

  }

  it should "predict probabilities" in {
    model.predictProba(Seq(20L, 22L)).sum should equal(1.0)
    model.predictProba(Seq(22L)).sum should equal(1.0)
    model.predictProba(Seq(21L)).sum should equal(1.0)
    model.predictProba(Seq(20L, 45L, 22L)).sum should equal(1.0)
    model.predictProba(Seq(30L)).sum should equal(1.0)
    model.classes should have size(2L)
    model.predictProba(Seq(30L)) should have size(2L)
    model.predictProba(Seq(20L)).sum should equal(1.0)
  }

  "On Mushroom" should "extract 137 rules, with sup=3000 and conf=0.5" in {
    val inputFile = "./src/test/resources/mushroom.dat"
    val all = sc.textFile(inputFile)
    val transactions = all.map(_.split(" ").map(_.toLong)).collect().map {
      x => val (t0, t1) = x.partition(_ < 3)
        (t1, t0.head)
    }
    val l3 = new L3(numClasses = 3, minSupport = 0.369, minChi2 = 0, strategy = "support") //they start from 1, minsup=3000

    val model=l3.train(transactions)

    model.rules should have size 137
  }

  "The L3 version 2 (with information gain) rules extractor" should "extract rules" in {
    new L3(numClasses = 2, minChi2 = 0.0, minSupport = 0.1, strategy = "gain").train(List[(Array[Long], Long)](
      (Array(14, 13, 11, 10), 0),
      (Array(13, 12, 10), 1),
      (Array(14, 13, 11, 10), 0),
      (Array(14, 13, 12, 10), 1),
      (Array(14, 13, 12, 11, 10), 0),
      (Array(13, 12, 11), 1)
    )).toString().split("\n") should equal(
      """14 11 -> 0 (0.500000, 1.000000, 6.000000)
        |12 -> 1 (0.500000, 0.750000, 3.000000)""".stripMargin.split("\n"))
  }

  private val ex2: List[(Array[Long], Long)] = List[(Array[Long], Long)](
    (Array(24, 20, 21, 23), 0),
    (Array(20, 22, 23), 1),
    (Array(24, 20, 21, 23), 0),
    (Array(24, 20, 22, 23), 1),
    (Array(24, 20, 22, 21, 23), 0),
    (Array(20, 22, 21), 1),
    (Array(24, 21, 23), 1)
  )

  private val ex3: List[(Array[String], Long)] = List[(Array[String], Long)](
    (Array("A", "B", "C", "D"), 0),
    (Array("A", "B", "D"), 1)
  )

//  ignore should "work with string items" in {
//    new L3(numClasses = 2, minChi2 = 0.0, minSupport = 0.1, minConfidence = 0.3).train(ex3).toString().split("\n") should equal(
//      """boh"""
//    )
//  }

  it should "extract rules (2)" in {
    new L3(numClasses = 2, minChi2 = 0.0, minSupport = 0.1, minConfidence = 0.3, strategy = "gain").train(ex2).toString().split("\n") should equal(
      """24 -> 0 (0.428571, 0.600000, 2.100000)
        |22 21 -> 1 (0.142857, 0.500000, 0.058333)""".stripMargin.split("\n"))
  }

  it should "filter rules" in {
    new L3(numClasses = 2, minChi2 = 2.0, minSupport = 0.2, minConfidence = 0.6, strategy = "gain").train(ex2).toString().split("\n") should equal(
      """24 -> 0 (0.428571, 0.600000, 2.100000)""".stripMargin.split("\n"))
  }
  it should "filter rules by support" in {
    /*N.B.: rule 22 20 -> 1 becomes 22 -> with the higher minSupport*/
    new L3(numClasses = 2, minChi2 = 0.0, minSupport = 0.2, minConfidence = 0.3, strategy = "gain").train(ex2).toString().split("\n") should equal(
      """22 -> 1 (0.428571, 0.750000, 1.215278)
        |24 -> 0 (0.428571, 0.600000, 2.100000)""".stripMargin.split("\n"))
  }
  it should "filter rules by confidence" in {
    new L3(numClasses = 2, minChi2 = 0.0, minSupport = 0.1, minConfidence = 0.7, strategy = "gain").train(ex2).toString().split("\n") should equal(
      """22 -> 1 (0.428571, 0.750000, 1.215278)""".stripMargin.split("\n"))
  }
  it should "filter rules by chi2" in {
    new L3(numClasses = 2, minChi2 = 2.0, minSupport = 0.1, minConfidence = 0.3, strategy = "gain").train(ex2).toString().split("\n") should equal(
      """24 -> 0 (0.428571, 0.600000, 2.100000)""".stripMargin.split("\n"))
  }

}

