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

package it.polito.dbdmg.spark.mllib.fpm

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag

/**
  * FP-Tree data structure used in FP-Growth.
  * @tparam T item type
  */
private[fpm] class FPTree[T, ClassType](val classCount: scala.collection.immutable.Map[ClassType, Int] = Map.empty)
                                       (implicit c: ClassTag[T]) extends Serializable {

  import it.polito.dbdmg.spark.mllib.fpm.FPTree._

  val root: Node[T] = new Node(null, numClasses)
  val class2Index: scala.collection.immutable.Map[ClassType, Int] = classCount.keys.zipWithIndex.toMap
  val idx2class: scala.collection.immutable.Map[Int, ClassType] = class2Index.map(_.swap)
  def numClasses: Int = if (classCount == null) 0 else classCount.keys.size

  private val summaries: mutable.Map[T, Summary[T]] = mutable.Map.empty

  /** Adds a transaction with classes count. */
  def add(t: Iterable[T], classesCount: mutable.ArrayBuffer[Long]): this.type = {
    var curr = root
    for (i <- (0 to curr.classesCount.size-1)) {
      curr.classesCount(i) += classesCount(i)
    }
    t.foreach { item =>
      val summary = summaries.getOrElseUpdate(item, new Summary(numClasses))
      for (i <- (0 to summary.classesCount.size-1)) {
        summary.classesCount(i) += classesCount(i)
      }
      val child = curr.children.getOrElseUpdate(item, {
        val newNode = new Node(curr, numClasses)
        newNode.item = item
        summary.nodes += newNode
        newNode
      })
      for (i <- (0 to child.classesCount.size-1)) {
        child.classesCount(i) += classesCount(i)
      }
      curr = child
    }
    this
  }

  /** Adds a transaction with count and class Label. */
  def addAndCountClasses(t: (Array[T], ClassType), count: Long = 1L): this.type = {
    require(count > 0)
    var curr = root
    curr.classesCount(class2Index(t._2)) += 1L
    t._1.foreach { item =>
      val summary = summaries.getOrElseUpdate(item, new Summary(numClasses))
      summary.classesCount(class2Index(t._2)) += 1L
      val child = curr.children.getOrElseUpdate(item, {
        val newNode = new Node(curr, numClasses)
        newNode.item = item
        summary.nodes += newNode
        newNode
      })
      child.classesCount(class2Index(t._2)) += 1L
      curr = child
    }
    this
  }

  /** Merges another FP-Tree. */
  def merge(other: FPTree[T, ClassType]): this.type = {
    other.transactions.foreach { case (t, c) =>
      add(t, c)
    }
    this
  }

  /** Gets a subtree with the suffix. */
  private def project(suffix: T): FPTree[T, ClassType] = {
    val tree = new FPTree[T, ClassType](classCount)
    if (summaries.contains(suffix)) {
      val summary = summaries(suffix)
      summary.nodes.foreach { node =>
        var t = List.empty[T]
        var curr = node.parent
        while (!curr.isRoot) {
          t = curr.item :: t
          curr = curr.parent
        }
        tree.add(t, node.classesCount)
      }
    }
    tree
  }

  /** Returns all transactions in an iterator. */
  def transactions: Iterator[(List[T], mutable.ArrayBuffer[Long])] = getTransactions(root)

  /** Returns all transactions under this node. */
  private def getTransactions(node: Node[T]): Iterator[(List[T], mutable.ArrayBuffer[Long])] = {
    node.children.iterator.flatMap { case (item, child) =>
      getTransactions(child).map { case (t, c) =>
        for (i <- (0 to node.classesCount.size-1)) {
          node.classesCount(i) -= c(i)
        }
        (item :: t, c)
      }
    } ++ {
      if (node.classesCount.sum > 0) {
        Iterator.single((Nil, node.classesCount))
      } else {
        Iterator.empty
      }
    }
  }

  /** Extracts all patterns with valid suffix and minimum count. */
  def extractAssocRules(
                         minCount: Long,
                         maxLength: Int,
                         minConfidence: Double = 0.5,
                         minChi2: Double = 3.841,
                         inputCount: Int,
                         validateSuffix: T => Boolean = _ => true
                       ): Iterator[(List[T], ClassType, Double, Double, Double)] = {

    summaries.iterator.flatMap { case (item, summary) =>
      val supClasses = summary.classesCount
      val maxCount = supClasses.max
      if (maxLength > 0 && validateSuffix(item) && maxCount >= minCount) {
        var it: Iterator[(List[T], ClassType, Double, Double, Double)] = Iterator.empty
        val labels = supClasses.zipWithIndex.filter(_._1 == maxCount).map(x => idx2class(x._2))

        for (label <- labels) {
          val supCons = classCount(label).toDouble / inputCount
          val sup = maxCount.toDouble / inputCount
          val supAnt = supClasses.sum.toDouble / inputCount
          val conf = sup.toDouble / supAnt
          //val supCons = root.classesCount(class2Index(label)).toDouble / root.classesCount.sum
          val chi2 = inputCount * {
            List(sup, supAnt - sup, supCons - sup, 1 - supAnt - supCons + sup) zip
              List(supAnt * supCons, supAnt * (1 - supCons), (1 - supAnt) * supCons, (1 - supAnt) * (1 - supCons)) map
              { case (observed: Double, expected: Double) => math.pow((observed - expected), 2) / expected } sum
          }
          it = it ++ Iterator((item :: Nil, label, sup, conf, chi2))
        }

        it ++ project(item).extractAssocRules(minCount, maxLength-1,
          minConfidence, minChi2, inputCount).map { case (t, label, sup, conf, chi2) =>
          ((item :: t, label, sup, conf, chi2))
        }.filter { case (t, label, sup, conf, chi2) =>
          (conf >= minConfidence && (chi2 >= minChi2 || chi2.isNaN))
        }
      } else {
        Iterator.empty
      }
    }
  }

}

private[fpm] object FPTree {

  /** Representing a node in an FP-Tree. */
  class Node[T](val parent: Node[T], val numClasses: Int = 2) extends Serializable {
    var item: T = _
    val children: mutable.Map[T, Node[T]] = mutable.Map.empty
    val classesCount: mutable.ArrayBuffer[Long] = mutable.ArrayBuffer.fill(numClasses)(0L)

    def isRoot: Boolean = parent == null
  }

  /** Summary of a item in an FP-Tree. */
  private class Summary[T](val numClasses: Int = 2) extends Serializable {
    val classesCount: mutable.ArrayBuffer[Long] = mutable.ArrayBuffer.fill(numClasses)(0L)
    val nodes: ListBuffer[Node[T]] = ListBuffer.empty
  }
}
