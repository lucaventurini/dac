import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD

/**
 * Created by luca on 24/02/15.
 */

case class Rule(antecedent:Set[Long], consequent:Long, support:BigInt, confidence:Double) {
  override def toString() = {
    antecedent.mkString(" ") +" -> " + consequent + " " + "(" + support + "," + confidence + ")"
  }
}

class L3 (val numClasses:Int) extends java.io.Serializable{

  def train(input: RDD[Array[Long]]): L3Model = {

    val fpg = new FPGrowth()
      .setMinSupport(0.2)
      .setNumPartitions(10)
    val model = fpg.run(input)

    model.freqItemsets.map{case (items, sup) => (items.partition(_ < numClasses), sup)}

    val antecedents = model.freqItemsets.map{case (items, sup) => val x = items.partition(_ < numClasses);(x._2.toSet,(x._1, sup))}
    val supAnts = antecedents.map(x => (x._1, x._2._2)).reduceByKey(_ + _)
    val rules = antecedents.filter(_._2._1.nonEmpty).join(supAnts).map{
      case (ant, ((classLabels, sup), supAnt)) => Rule(ant, classLabels(0), sup, sup/supAnt.toDouble)
    }

    new L3Model(rules)
  }

  def train[Item](input: RDD[Array[Item]], isClassLabel:(Item => Boolean)) {
    //TODO: generalize to any type of item representation, given a function to distinguish a class label
    //default: Item = Long, isClassLabel = (x => x < numClasses)
  }

}
