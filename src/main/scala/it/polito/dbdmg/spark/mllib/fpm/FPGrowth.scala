package it.polito.dbdmg.spark.mllib.fpm

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



import java.lang.{Iterable => JavaIterable}
import java.{util => ju}

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, Partitioner, SparkException}
//import org.apache.spark.Logging
//import org.apache.spark.internal.Logging
import org.apache.spark.mllib.tree.impurity.Gini

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag
import it.polito.dbdmg.ml.Rule
import org.apache.log4j.Logger
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.{ArrayBuffer, HashMap}


object Holder extends Serializable {
//  @transient lazy val log = Logger.getLogger(getClass.getName)
  @transient lazy val log = org.apache.log4j.LogManager.getLogger("myLogger")
}

/**
  * :: Experimental ::
  *
  * Model trained by [[FPGrowth]], which holds frequent itemsets.
  * @param freqItemsets frequent itemset, which is an RDD of [[FreqItemset]]
  * @tparam Item item type
  */
@Experimental
class FPGrowthModel[Item: ClassTag](val freqItemsets: Iterable[FreqItemset[Item]]) extends Serializable

/**
  * :: Experimental ::
  *
  * A parallel FP-growth algorithm to mine frequent itemsets. The algorithm is described in
  * [[http://dx.doi.org/10.1145/1454008.1454027 Li et al., PFP: Parallel FP-Growth for Query
  *  Recommendation]]. PFP distributes computation in such a way that each worker executes an
  * independent group of mining tasks. The FP-Growth algorithm is described in
  * [[http://dx.doi.org/10.1145/335191.335372 Han et al., Mining frequent patterns without candidate
  *  generation]].
  *
  * @param minSupport the minimal support level of the frequent pattern, any pattern appears
  *                   more than (minSupport * size-of-the-dataset) times will be output
  * @param numPartitions number of partitions used by parallel FP-growth
  *
  * @see [[http://en.wikipedia.org/wiki/Association_rule_learning Association rule learning
  *       (Wikipedia)]]
  */
@Experimental
class FPGrowth[Item] private (
     private var minSupport: Double,
     private var numPartitions: Int,
     private var strategy: String = "gain",
     private var minInfoGain: Double = 0.0,
     private var minConfidence: Double = 0.5,
     private var minChi2: Double = 3.841,
     private var rulesMaxLen: Int = 10) extends Logging with Serializable {

  /**
    * Constructs a default instance with default parameters {minSupport: `0.3`, numPartitions: same
    * as the input data}.
    */
  def this() = this(0.3, -1)

  /**
    * Sets the minimal support level (default: `0.3`).
    */
  def setMinSupport(minSupport: Double): this.type = {
    this.minSupport = minSupport
    this
  }

  /**
    * Sets the number of partitions used by parallel FP-growth (default: same as input data).
    */
  def setNumPartitions(numPartitions: Int): this.type = {
    this.numPartitions = numPartitions
    this
  }

  def setMinConfidence(minConfidence: Double): this.type = {
    this.minConfidence = minConfidence
    this
  }

  def setMinChi2(minChi2: Double): this.type = {
    this.minChi2 = minChi2
    this
  }

  def setMinInfoGain(minInfoGain: Double): this.type = {
    this.minInfoGain = minInfoGain
    this
  }

  def setStrategy(strategy: String): this.type = {
    this.strategy = strategy
    this
  }

  def setRulesMaxLength(rulesMaxLen: Int): this.type = {
    this.rulesMaxLen = rulesMaxLen
    this
  }

  def withInformationGain(): Boolean = {
    if (strategy.equals("gain")) true else false
  }

  /**
    * Computes an Associative Classifier that contains frequent itemsets with
    * a class label, based on FP-Growth
    * @param data input data set, each element contains a transaction with its class label
    * @param classCount class count if the entire dataset, each element has a key that is a class label
    *                   and a value that is the count of such class label in the entire data set
    * @tparam Item class type if the transaction and the class label
    * @return an Iterable of rules
    */
  def run[Item: ClassTag](data: Iterable[(Array[Item], Item)],
      classCount: scala.collection.immutable.Map[Item, Int]): Iterable[Rule[Item]] = {
    val count = data.size
    val minCount = math.ceil(minSupport * count).toLong
    val numParts = 1
    val partitioner = new HashPartitioner(numParts)
    if (withInformationGain()) {
      //val selected = featureSelect(data, classCount, count)
      val items = genGainItems(data, minCount, partitioner, classCount, count)
      //val items = genFreqItems(data, minCount, partitioner)
      genAssocRulesWInfoGain(data, minCount, items, partitioner, classCount, count)
    } else {
      val freqItems = genFreqItems(data, minCount, partitioner)
      genAssocRules(data, minCount, freqItems, partitioner, classCount, count)
    }
  }

  /**
    * Computes a feature selection on input data set
    * @param data input data set, each element contains a transaction with its class label
    * @param classCount class count if the entire dataset, each element has a key that is a class label
    *                   and a value that is the count of such class label in the entire data set
    * @param inputCount the total count of the input data set
    * @tparam Item class type if the transaction and the class label
    * @return the transactions modified in order to contain only the fatures selected
    */
  def featureSelect[Item: ClassTag](data: Iterable[(Array[Item], Item)],
                                    classCount: scala.collection.immutable.Map[Item, Int],
                                    inputCount: Int): Iterable[(Array[Item], Item)] = {
    val giniFather = Gini.calculate(classCount.map(_._2.toDouble).toArray, inputCount.toDouble)
    val fs = data.flatMap(x => x._1.zipWithIndex.map((_,x._2)))
      .groupBy(x => x._1._2).mapValues {
      x => x.groupBy(_._2).mapValues(_.size)
    }.mapValues { x =>
      val omega = x.map(_._2).sum.toDouble / inputCount
      val giniSon = Gini.calculate(x
        .map(_._2.toDouble)
        .toArray,x.map(_._2)
        .sum.toDouble)
      (omega * (giniFather - giniSon))
    }.toArray
      .sortBy(-_._2)
      .zipWithIndex
      .filter(x => x._2 < 10)  // here you can select only the number of features to select
      .map(_._1._1)

    data.map{ x =>
      (x._1.zipWithIndex.filter(i => fs.contains(i._2)).map(_._1), x._2)
    }
  }

  /**
    * Generates frequent items by filtering the input data using minimal support level.
    * @param minCount minimum count for frequent itemsets
    * @param partitioner partitioner used to distribute items
    * @return array of frequent pattern ordered by their frequencies
    */
  private def genFreqItems[Item: ClassTag](
                                            data: Iterable[(Array[Item], Item)],
                                            minCount: Long,
                                            partitioner: Partitioner): Array[Item] = {
    data.map(_._1).flatMap { t =>
      val uniq = t.toSet
      if (t.size != uniq.size) {
        throw new SparkException(s"Items in a transaction must be unique but got ${t.toSeq}.")
      }
      t
    }.groupBy(x => x).mapValues(_.size)
      .filter(_._2 >= minCount)
      .toArray
      .sortBy(-_._2)
      .map(_._1)
  }

  /**
    * Generates most significant items by filtering the input data using minimal support level, ordering
    * them using the Information Gain value associated to each item.
    * @param minCount minimum count for frequent itemsets
    * @param partitioner partitioner used to distribute items
    * @return array of frequent pattern ordered by their information gain
    */
  private def genGainItems[Item: ClassTag](
                                            data: Iterable[(Array[Item], Item)],
                                            minCount: Long,
                                            partitioner: Partitioner,
                                            classCount: scala.collection.immutable.Map[Item, Int],
                                            inputCount: Long): Array[Item] = {
    val giniFather = Gini.calculate(classCount.map(_._2.toDouble).toArray, inputCount.toDouble)
    def countByLabel[T <: (Item, Item)](xs: TraversableOnce[T]): Map[Item, Iterable[Int]] = {
      xs.foldLeft(HashMap.empty[Item, mutable.Map[Item,Int]])(
        (acc, x) => {
          if(!acc.contains(x._1))
            acc(x._1) = mutable.HashMap.empty[Item,Int].withDefaultValue(0)
          acc(x._1)(x._2) += 1
          acc}).mapValues(_.values).toMap
    }
    val itemAndLabels: Iterable[(Item, Item)] = data.flatMap { case (items, label) =>
//      val uniq = items.toSet
//      if (items.length != uniq.size) {
//        throw new SparkException(s"Items in a transaction must be unique but got ${items.toSeq}.")
//      }
      items.map {
        i => (i, label)
      }
    }
    Holder.log.info("FP-Growth: starting counting items...")

    /*    val itemCountsByLabel: Map[Item, Iterable[Int]] = data.flatMap { case (items, label) =>
          val uniq = items.toSet
          if (items.size != uniq.size) {
            throw new SparkException(s"Items in a transaction must be unique but got ${items.toSeq}.")
          }
          items.map {
            i => (i, label)
          }
        }
          .groupBy(x => x._1)
          .filter(_._2.size >= minCount)
          .mapValues {
            x => x.groupBy(_._2).mapValues(_.size).values
          }*/
    val itemCountsByLabel = countByLabel(itemAndLabels).filter(_._2.sum >= minCount)
    Holder.log.info("FP-Growth: items counted")
    itemCountsByLabel
      .map { case (item, count) =>
      val omega = count.sum.toDouble / inputCount
      val giniSon = Gini.calculate(count
        .map(_.toDouble)
        .toArray,count
        .sum.toDouble)
      (item, omega * (giniFather - giniSon))
    }//.filter(_._2 >= minInfoGain)
      .filter(_._2 > 0)
      .toArray
      .sortBy(-_._2)
      .map(_._1)
//      .take(10000)

      /*
         alternative version: you can evaluate which one is better
         this version group by the same items, then filters out the items without the
         minimum support threshold, then for each item create the count of the classes starting
         form the labels contained in classCount map, which contains all the labels
         of the dataset with their complete count. Please note that values of the map
         are not considered here, only the keys (e.g. the labels) are taken from the map.
         This one has one group by operation less, probably this version is faster.
          */
      /*.groupBy(x => x._1)
      .filter(_._2.size >= minCount)
      .mapValues { values =>
        classCount.keys.map { label =>
          values.filter{ case (item, itemLabel) =>  itemLabel == label }.size
        }
      }.map { case (item, classesCount) =>
      val omega = classesCount.sum.toDouble / inputCount
      val giniSon = Gini.calculate(classesCount
        .map(_.toDouble)
        .toArray, classesCount.sum.toDouble)
      (item, omega * (giniFather - giniSon))
    }//.filter(_._2 > minInfoGain)
      .toArray
      .sortBy(-_._2)
      .map(_._1)*/
  }

  /**
    * Generate frequent rule sets by building FP-Trees, the extraction is done on each partition.
    * @param data transactions
    * @param minCount minimum count for frequent itemsets
    * @param freqItems frequent items
    * @param partitioner partitioner used to distribute transactions
    * @return an Iterable of rules
    */
  private def genAssocRules[Item: ClassTag](
                                           data: Iterable[(Array[Item], Item)],
                                           minCount: Long,
                                           freqItems: Array[Item],
                                           partitioner: Partitioner,
                                           classCount: scala.collection.immutable.Map[Item, Int],
                                           inputCount: Int): Iterable[Rule[Item]] = {
    val itemToRank = freqItems.zipWithIndex.toMap
    data.flatMap { transaction =>
      genCondTransactions(transaction, itemToRank, partitioner)
    }.aggregate(new FPTree[Int, Item](classCount))(
      (tree, transaction) => tree.addAndCountClasses(transaction._2, 1L),
      (tree1, tree2) => tree1.merge(tree2))
      .extractAssocRules(minCount, rulesMaxLen, minConfidence, minChi2, inputCount)
      .map { case (ranks, label, sup, conf, chi2) =>
        new Rule[Item](ranks.map(i => freqItems(i)).toArray, label, sup, conf, chi2)
      }.toIterable
  }

  /**
    * Generate most significant rule sets by building FP-Trees, the extraction is done on each partition.
    * @param data transactions
    * @param minCount minimum count for frequent itemsets
    * @param items frequent items ordered by most significant information gain
    * @param partitioner partitioner used to distribute transactions
    * @param sorted imply that itmes in transactions are yet ordered by their information gain values
    * @return an Iterable of rules
    */
  private def genAssocRulesWInfoGain[Item: ClassTag](
                                             data: Iterable[(Array[Item], Item)],
                                             minCount: Long,
                                             items: Array[Item],
                                             partitioner: Partitioner,
                                             classCount: scala.collection.immutable.Map[Item, Int],
                                             inputCount: Int,
                                             sorted: Boolean = false): Iterable[Rule[Item]] = {
    if (sorted) {
      data.aggregate(new FPTree[Item, Item](classCount))(
        (tree, transaction) => tree.addAndCountClasses(transaction, 1L),
        (tree1, tree2) => tree1.merge(tree2))
        .extractAssocRulesWInfoGain(minCount, 10, minConfidence, minChi2, inputCount)
        .map { case (ant, label, sup, conf, chi2) =>
          new Rule[Item](ant.toArray, label, sup, conf, chi2)
        }.toIterable
    } else {
      val itemToRank = items.zipWithIndex.toMap
      data.flatMap { transaction =>
        genCondTransactions(transaction, itemToRank, partitioner)
      }.aggregate(new FPTree[Int, Item](classCount))(
        (tree, transaction) => tree.addAndCountClasses(transaction._2, 1L),
        (tree1, tree2) => tree1.merge(tree2))
        .extractAssocRulesWInfoGain(minCount, rulesMaxLen, minConfidence, minChi2, inputCount)
        .map { case (ranks, label, sup, conf, chi2) =>
          new Rule[Item](ranks.map(i => items(i)).toArray, label, sup, conf, chi2)
        }.toIterable
    }
  }



  /**
    * Generates conditional transactions.
    * @param transaction a transaction
    * @param itemToRank map from item to their rank
    * @param partitioner partitioner used to distribute transactions
    * @return a map of (target partition, conditional transaction)
    */
  private def genCondTransactions[Item: ClassTag](
     transaction: (Array[Item], Item),
     itemToRank: Map[Item, Int],
     partitioner: Partitioner): mutable.Map[Int, (Array[Int], Item)] = {
    val output = mutable.Map.empty[Int, (Array[Int], Item)]
    // Filter the basket by frequent items pattern and sort their ranks.
    val filtered = (transaction._1.flatMap(itemToRank.get), transaction._2)
    ju.Arrays.sort(filtered._1)
    val n = filtered._1.length
    var i = n - 1
    while (i >= 0) {
      val item = filtered._1(i)
      val part = partitioner.getPartition(item)
      if (!output.contains(part)) { //todo: eliminate partitions
        output(part) = (filtered._1.slice(0, i + 1), filtered._2)
      }
      i -= 1
    }
    output
  }

}

/**
  * :: Experimental ::
  */
@Experimental
object FPGrowth {

  /**
    * Frequent itemset.
    * @param items items in this itemset. Java users should call [[FreqItemset#javaItems]] instead.
    * @param freq frequency
    * @tparam Item item type
    */
  class FreqItemset[Item](val items: Array[Item], val freq: Long) extends Serializable {

    /**
      * Returns items in a Java List.
      */
    def javaItems: java.util.List[Item] = {
      items.toList.asJava
    }
  }
}
