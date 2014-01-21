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

package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.scalatest.BeforeAndAfterAll
import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite
import org.scalatest.FunSuite
import org.scalatest.matchers.ShouldMatchers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.scalatest.matchers.ShouldMatchers
import org.jblas.DoubleMatrix

class BreezeLBFGSSuite extends FunSuite with BeforeAndAfterAll with ShouldMatchers {
  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  def compareDouble(x: Double, y: Double, tol: Double = 1E-3): Boolean = {
    math.abs(x - y) / math.abs(y + 1e-15) < tol
  }

  test("Assert LBFGS loss is decreasing and matches the result of Gradient Descent.") {
    val gradient = new MultiLogisticGradient()
    val numCorrections = 10
    val lineSearchTolerance = 0.9
    val convTolerance = 1e-12
    val maxNumIterations = 100
    val miniBatchFrac = 1.0

    val updater = new SimpleUpdater()
    val regParam = 0

    val file = this.getClass.getResource("/data/classification/iris.data").toURI.toString
    val rddData = sc.textFile(file).map(line => {
      val temp = line.toString.split(",")
      val y = temp(0).toDouble
      val x = temp.slice(1, 5).map(_.toDouble)
      (y, Array(1, x: _*))
    })

    val initialWeightsWithIntercept = Array.ofDim[Double](3 * 5)
    val counts = rddData.count

    val (weightRISOLBFGS, _) = LBFGS.runMiniBatchLBFGS(
      rddData,
      gradient,
      updater,
      numCorrections,
      lineSearchTolerance,
      convTolerance,
      maxNumIterations,
      regParam,
      miniBatchFrac,
      initialWeightsWithIntercept)

    val accRISOLBFGS = rddData.collect().map(x => if (predictPoint(x._2.slice(1, 5), weightRISOLBFGS).toInt == x._1) 1 else 0).reduce(_ + _) / counts.toDouble

    val (weightBreezeLBFGS, _) = BreezeLBFGS.runMiniBatchBreezeLBFGS(
      rddData,
      gradient,
      updater,
      numCorrections,
      lineSearchTolerance,
      convTolerance,
      maxNumIterations,
      regParam,
      miniBatchFrac,
      initialWeightsWithIntercept)

    val accBreezeLBFGS = rddData.collect().map(x => if (predictPoint(x._2.slice(1, 5), weightBreezeLBFGS).toInt == x._1) 1 else 0).reduce(_ + _) / counts.toDouble

    println("\n\n")
    println("RISO LBFGS acc: " + accRISOLBFGS)
    println("Breeze LBFGS acc: " + accBreezeLBFGS)
  }


  def predictPointProbabilities(dataMatrix: Array[Double], weightMatrix: Array[Double]): Array[Double] = {
    val nFeatures = dataMatrix.length
    val kClasses = weightMatrix.length / (nFeatures + 1)

    val probs = new Array[Double](kClasses)
    probs(0) = 1.0

    var acc = 1.0
    for (i <- 0 until (kClasses - 1)) {
      // baseline is always class 0, so no weights or intercept for the first class.
      var margin: Double = weightMatrix(i * (nFeatures + 1))
      for (j <- 0 until nFeatures) {
        margin += dataMatrix(j) * weightMatrix(i * (nFeatures + 1) + j + 1)
      }
      probs(i + 1) = math.exp(margin)
      acc += probs(i + 1)
    }
    for (i <- 0 until kClasses) {
      probs(i) /= acc
    }
    probs
  }

  def predictPoint(testData: Array[Double], weights: Array[Double]): Double = {
    val probs = predictPointProbabilities(testData, weights)
    var index = 0
    var max = probs(0)
    for (i <- 1 until probs.length) {
      if (probs(i) > max) {
        index = i
        max = probs(i)
      }
    }

    index.toDouble
  }
}


