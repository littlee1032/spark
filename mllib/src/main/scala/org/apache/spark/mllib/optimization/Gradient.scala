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

import org.jblas.DoubleMatrix

/**
 * Class used to compute the gradient for a loss function, given a single data point.
 */
abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data - Feature values for one data point. Column matrix of size dx1
   *               where d is the number of features.
   * @param label - Label for this data item.
   * @param weights - Column matrix containing weights for every feature.
   *
   * @return A tuple of 2 elements. The first element is a column matrix containing the computed
   *         gradient and the second element is the loss computed at this data point.
   *
   */
  def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix): 
      (DoubleMatrix, Double)
}

/**
 * Compute gradient and loss for a logistic loss function, as used in binary classification.
 * See also the documentation for the precise formulation.
 */
class LogisticGradient extends Gradient {
  override def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix): 
      (DoubleMatrix, Double) = {
    val margin: Double = -1.0 * data.dot(weights)
    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label

    val gradient = data.mul(gradientMultiplier)
    val loss =
      if (label > 0) {
        math.log(1 + math.exp(margin))
      } else {
        math.log(1 + math.exp(margin)) - margin
      }

    (gradient, loss)
  }
}

/**
 * Compute gradient and loss for a multinomial logistic regression loss function.
 * NOTE: This assumes that the labels are {0, 1, 2, ..., N - 1} for N classes
 * classification problem
 */
class MultiLogisticGradient extends Gradient {
  override def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix):
  (DoubleMatrix, Double) = {
    def alpha(i: Int): Int = if (i == 0) 1 else 0
    def delta(i: Int, j: Int): Int = if (i == j) 1 else 0

    val y = Math.round(label).toInt
    val gradient = new DoubleMatrix(weights.rows, 1)

    // Note: data.rows contains intercept.
    val classes: Int = (weights.rows / data.rows) + 1

    var denominator = 1.0
    val numerators: Array[Double] = Array.ofDim[Double](classes - 1)

    var i = 0; var j = 0
    while(i < classes - 1) {
      j = 0
      var acc = 0.0
      while(j < data.rows) {
        acc += data.get(j, 0) * weights.get(i * data.rows + j, 0)
        j += 1
      }
      numerators(i) = math.exp(acc)
      denominator += numerators(i)
      i += 1
    }

    i = 0
    while (i < weights.length) {
      val m: Int = i % data.rows
      val c: Int = (i - m) / data.rows
      gradient.put(i, 0, gradient.get(i,0) -
        ((1 - alpha(y)) * delta(y, c + 1) - numerators(c) / denominator) * data.get(m, 0))
      i += 1
    }

    val loss = - math.log((if(y == 0) 1.0 else numerators(y - 1)) / denominator)

    (gradient, loss)
  }
}

/**
 * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/n ||A weights-y||^2
 * See also the documentation for the precise formulation.
 */
class LeastSquaresGradient extends Gradient {
  override def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix): 
      (DoubleMatrix, Double) = {
    val diff: Double = data.dot(weights) - label

    val loss = diff * diff
    val gradient =  data.mul(2.0 * diff)

    (gradient, loss)
  }
}

/**
 * Compute gradient and loss for a Hinge loss function, as used in SVM binary classification.
 * See also the documentation for the precise formulation.
 * NOTE: This assumes that the labels are {0,1}
 */
class HingeGradient extends Gradient {
  override def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix):
      (DoubleMatrix, Double) = {

    val dotProduct = data.dot(weights)

    // Our loss function with {0, 1} labels is max(0, 1 - (2y – 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0

    if (1.0 > labelScaled * dotProduct) {
      (data.mul(-labelScaled), 1.0 - labelScaled * dotProduct)
    } else {
      (DoubleMatrix.zeros(1, weights.length), 0.0)
    }
  }
}