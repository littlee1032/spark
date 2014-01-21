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

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite
import org.scalatest.matchers.ShouldMatchers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint

class LBFGSSuite extends FunSuite with BeforeAndAfterAll with ShouldMatchers {
  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  test("Assert LBFGS loss is decreasing and match the result of Gradient Descent.") {
    val nPoints = 10000
    val A = 2.0
    val B = -1.5

    val initialB = -1.0
    val initialWeights = Array(initialB)

    val gradient = new LogisticGradient()
    val numCorrections = 10
    val lineSearchTolerance = 0.9
    val convTolerance = 1e-12
    val maxNumIterations = 10
    val regParam = 0
    val miniBatchFrac = 1.0

    // Add a extra variable consisting of all 1.0's for the intercept.
    val testData = GradientDescentSuite.generateGDInput(A, B, nPoints, 42)
    val data = testData.map { case LabeledPoint(label, features) =>
      label -> Array(1.0, features: _*)
    }

    val dataRDD = sc.parallelize(data, 2).cache()
    val initialWeightsWithIntercept = Array(1.0, initialWeights: _*)

    val (_, loss) = LBFGS.runMiniBatchLBFGS(
      dataRDD,
      gradient,
      numCorrections,
      lineSearchTolerance,
      convTolerance,
      maxNumIterations,
      regParam,
      miniBatchFrac,
      initialWeightsWithIntercept)

    assert(loss.last - loss.head < 0, "loss isn't decreasing.")

    val lossDiff = loss.init.zip(loss.tail).map {
      case (lhs, rhs) => lhs - rhs
    }
    assert(lossDiff.count(_ > 0).toDouble / lossDiff.size > 0.8)

    val updater = new SimpleUpdater()
    val stepSize = 1.0
    // Well, GD converges slower.
    val numGDIterations = 50
    val initialGDWeightsWithIntercept = Array(1.0, initialWeights: _*)
    val (_, lossGD) = GradientDescent.runMiniBatchSGD(
      dataRDD,
      gradient,
      updater,
      stepSize,
      numGDIterations,
      regParam,
      miniBatchFrac,
      initialGDWeightsWithIntercept)

    assert(Math.abs((lossGD.last - loss.last)/loss.last) < 0.01,"LBFGS should match GD result.")
  }
}