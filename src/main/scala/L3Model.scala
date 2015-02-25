import org.apache.spark.rdd.RDD

/**
 * Created by luca on 24/02/15.
 */
class L3Model(val rules:RDD[Rule]) {

  override def toString() = {
    rules.collect().map(_.toString).mkString("\n")
  }
}
