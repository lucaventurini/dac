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
      (p0, p1.head)
  }
  lazy val model = {new L3(numClasses = 2, minChi2 = 0.0).train2(data)}
  lazy val modelCovered = model.dBCoverage2(data)

  "The L3 rule extractor" should "extract rules" in {
    new L3(numClasses = 2, minChi2 = 0.0).train2(List[(Array[Long], Long)](
      (Array(10, 11, 12), 0),
      (Array(0, 11, 12), 0),
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
    model.dBCoverage2(l).toString.split("\n") should
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

  /*  it should "cover all the DB when minSupp=0 and minConf=0 and maxChi2=Inf" in {
      //N:B: this test should work on ANY data, theoretically
      val model = {new L3(numClasses = 2, minSupport = 0, minConfidence = 0).train(sc.parallelize(input))}
      val modelCovered = model.dBCoverage(sc.parallelize(input))

      modelCovered.predict(sc.parallelize(input).map(_.toSet)).collect().forall(_ nonEmpty) should be(true)
    }*/

  /*it should "use training dataset by default" in {
    modelCovered.toString().split("\n") should equal(model.dBCoverage2().toString().split("\n"))
  }*/

  lazy val modelbag = new L3Ensemble(numClasses = 2, numModels = 2).train(sc.parallelize(Seq.fill(500)(Array(math.round(Random.nextFloat()).toLong, 2+(10*Random.nextDouble()).round, 13, 14))))

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
    val l3 = new L3(numClasses = 3, minSupport = 0.369, minChi2 = 0) //they start from 1, minsup=3000

    val model=l3.train2(transactions)

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
  lazy val model:L3LocalModel = {new L3(numClasses = 2, minChi2 = 0.0).train2(input)}

  lazy val modelCovered = model.dBCoverage2(input)

  "The L3 Local rule extractor" should "extract rules" in {
    new L3(numClasses = 2, minChi2 = 0.0).train2(List[(Array[Long], Long)](
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
    model.dBCoverage2(List((Array[Long](20, 21, 22),1L))).toString.split("\n") should
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
    model.dBCoverage2(List((Array[Long](20, 21, 22), 1L)), saveSpare = false).rulesIIlevel should have size 0
  }


  lazy val modelbag = new L3Ensemble(numClasses = 2, numModels = 2).train(sc.parallelize(Seq.fill(500)(Array(math.round(Random.nextFloat()).toLong, 2+(10*Random.nextDouble()).round, 13, 14))))

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

  "On Mushroom" should "extract 137 rules, with sup=3000 and conf=0.5" in {
    val inputFile = "./src/test/resources/mushroom.dat"
    val all = sc.textFile(inputFile)
    val transactions = all.map(_.split(" ").map(_.toLong)).collect().map {
      x => val (t0, t1) = x.partition(_ < 3)
        (t1, t0.head)
    }
    val l3 = new L3(numClasses = 3, minSupport = 0.369, minChi2 = 0) //they start from 1, minsup=3000

    val model=l3.train2(transactions)

    model.rules should have size 137
  }

}

