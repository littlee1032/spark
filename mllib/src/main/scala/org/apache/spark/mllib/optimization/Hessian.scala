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
 * Class used to compute the hessian for a loss function, given a single data point.
 * Since the dimension of Hessian will be square of the dimension of gradient, we only
 * provide addInPlace interface to reduce the overhead of creating new objects.
 */
abstract class Hessian extends Serializable {
  /**
   * Compute the Hessian given features of a single data point, and add back to input.
   * The result of hessian will be added to input field-hessian in place to reduce the overhead
   * of creating new object. Although it has side effect, but it can improve the performance dramatically.
   *
   * @param data - Feature values for one data point. Column matrix of size nx1
   *               where n is the number of features.
   * @param label - Label for this data item.
   * @param weights - Column matrix containing weights for every feature.
   * @param hessian - The new gradient will be added to this input in place.
   *
   */
  def computeAddInPlace(data: DoubleMatrix, label: Double, weights: DoubleMatrix, hessian: DoubleMatrix)
}