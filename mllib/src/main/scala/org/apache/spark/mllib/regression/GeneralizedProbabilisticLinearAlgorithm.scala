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

package org.apache.spark.mllib.regression

import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix

abstract class GeneralizedProbabilisticLinearAlgorithm(override val weights: Array[Double], override val intercept: Double)
  extends GeneralizedLinearModel(weights, intercept) {

  /**
   * Predict the probabilities given a data point and the weights learned.
   *
   * @param dataMatrix Row vector containing the features for this data point
   * @param weightMatrix Column vector containing the weights of the model
   * @param intercept Intercept of the model.
   */
  def predictPointProbabilities(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
                                intercept: Double): Array[Double]

  /**
   * Predict probabilities of each class for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted
   * @return RDD of Array[Double] where each entry contains the corresponding prediction
   */
  def predictProbabilities(testData: RDD[Array[Double]]): RDD[Array[Double]] = {
    // A small optimization to avoid serializing the entire model. Only the weightsMatrix
    // and intercept is needed.
    val localWeights = weightsMatrix
    val localIntercept = intercept

    testData.map { x =>
      val dataMatrix = new DoubleMatrix(1, x.length, x:_*)
      predictPointProbabilities(dataMatrix, localWeights, localIntercept)
    }
  }

  /**
   * Predict probabilities of each class for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return Array[Double] prediction from the trained model
   */
  def predictProbabilities(testData: Array[Double]): Array[Double] = {
    val dataMatrix = new DoubleMatrix(1, testData.length, testData: _*)
    predictPointProbabilities(dataMatrix, weightsMatrix, intercept)
  }

  /**
   * Predict the result given a data point and the weights learned.
   *
   * @param dataMatrix Row vector containing the features for this data point
   * @param weightMatrix Column vector containing the weights of the model
   * @param intercept Intercept of the model.
   */
  def predictPoint(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix, intercept: Double) = {
    val p = predictPointProbabilities(dataMatrix, weightMatrix, intercept)
    var index = 0
    var max = p(0)
    for (i <- 1 until p.length) {
      if (p(i) > max) {
        index = i
        max = p(i)
      }
    }
    index.toDouble
  }
}
