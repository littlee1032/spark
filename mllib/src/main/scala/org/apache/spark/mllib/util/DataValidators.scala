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

package org.apache.spark.mllib.util

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * A collection of methods used to validate data before applying ML algorithms.
 */
object DataValidators extends Logging {

  /**
   * Function to check if labels used for classification are either zero or one.
   *
   * @param data - input data set that needs to be checked
   *
   * @return True if labels are all zero or one, false otherwise.
   */
   val classificationLabels: RDD[LabeledPoint] => Boolean = { data =>
    val numInvalid = data.filter(x => Math.round(x.label) < 0 || Math.round(x.label) > 1).count()
    if (numInvalid != 0) {
      logError("Classification labels should be 0 or 1. Found " + numInvalid + " invalid labels")
    }
    numInvalid == 0
  }

  /**
   * Function to check if labels used for multinomial classification are
   * in Range(0, ..., K-1) for K classes classification.
   *
   * @param data - input data set that needs to be checked
   *
   * @return True if labels are all zero or one, false otherwise.
   */
  val multiClassificationLabels: Int => RDD[LabeledPoint] => Boolean = { classes => data =>
    val numInvalid = data.filter(x => Math.round(x.label) < 0 || Math.round(x.label) > (classes - 1)).count()
    if (numInvalid != 0) {
      logError("Classification labels should be in Range(0, ..., " + (classes - 1) + "). Found " + numInvalid + " invalid labels")
    }
    numInvalid == 0
  }
}
