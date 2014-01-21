///*
// * Licensed to the Apache Software Foundation (ASF) under one or more
// * contributor license agreements.  See the NOTICE file distributed with
// * this work for additional information regarding copyright ownership.
// * The ASF licenses this file to You under the Apache License, Version 2.0
// * (the "License"); you may not use this file except in compliance with
// * the License.  You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package org.apache.spark.mllib.classification
//
//import org.apache.spark.mllib.regression.{LabeledPoint, GeneralizedLinearAlgorithm, GeneralizedProbabilisticLinearAlgorithm}
//import org.apache.spark.{SparkContext, Logging}
//import org.jblas.DoubleMatrix
//import org.apache.spark.mllib.optimization._
//import org.apache.spark.mllib.util.{MLUtils, DataValidators}
//import org.apache.spark.rdd.RDD
//import org.apache.spark.mllib.regression.LabeledPoint
//
//
///**
// * Classification model trained using Multinomial Logistic Regression.
// *
// * @param weights Weights (including intercepts) computed for every feature.
// *                Since for multi logistic regression, we use first class (0) as baseline,
// *                there is no intercept and weights for first class.
// *                For a system with K classes (ranging from 0 to K-1), and N features (ranging from 0 to N-1),
// *                there will be (K - 1)*(N + 1) parameters.
// *
// *                The weights is define by,
// *                Array(    b1,     w10, w12, ...     w1(N-1),
// *                .,       .,   ., ...           .,
// *                b(K-1), w(K-1)1,   ., ... W(K-1)(N-1))
// *                where bk is the intercept for class-k, and Wkn is the weights for class-k and feature-n
// * @param classes The numbers of classes in the dependent variable.
// */
//class MultiLogisticRegressionModel(override val weights: Array[Double], val classes: Int)
//  extends GeneralizedProbabilisticLinearAlgorithm(weights, 0.0)
//  with ClassificationModel with Serializable {
//
//  override def predictPointProbabilities(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
//                                         intercept: Double): Array[Double] = {
//    val kClasses = classes
//    val nFeatures = (weightMatrix.rows / (kClasses - 1)) - 1
//
//    val probs = new Array[Double](kClasses)
//    probs(0) = 1.0
//
//    var acc: Double = 1
//    for (i <- 0 until (kClasses - 1)) {
//      // baseline is always class 0, so no weights or intercept for the first class.
//      var margin: Double = weightMatrix.get(i * (nFeatures + 1), 0)
//      for (j <- 0 until nFeatures) {
//        margin += dataMatrix.get(0, j) * weightMatrix.get(i * (nFeatures + 1) + j + 1, 0)
//      }
//      probs(i + 1) = math.exp(margin)
//      acc += probs(i + 1)
//    }
//    for (i <- 0 until kClasses) {
//      probs(i) /= acc
//    }
//
//    probs
//  }
//}
//
/////**
//// * Train a classification model for Logistic Regression using Stochastic Gradient Descent.
//// * NOTE: Labels used in Logistic Regression should be {0, 1}
//// */
////class LogisticRegressionWithSGD private (
////                                          var stepSize: Double,
////                                          var numIterations: Int,
////                                          var regParam: Double,
////                                          var miniBatchFraction: Double)
////  extends GeneralizedLinearAlgorithm[LogisticRegressionModel]
////  with Serializable {
////
////  val gradient = new LogisticGradient()
////  val updater = new SimpleUpdater()
////  override val optimizer = new GradientDescent(gradient, updater)
////    .setStepSize(stepSize)
////    .setNumIterations(numIterations)
////    .setRegParam(regParam)
////    .setMiniBatchFraction(miniBatchFraction)
////  override val validators = List(DataValidators.classificationLabels)
////
////  /**
////   * Construct a LogisticRegression object with default parameters
////   */
////  def this() = this(1.0, 100, 0.0, 1.0)
////
////  def createModel(weights: Array[Double], intercept: Double) = {
////    new LogisticRegressionModel(weights, intercept)
////  }
////}
//
//
//
//
///**
//* Compute gradient and loss for a multinomial logistic loss function.
//*/
//object MultiLogisticGradient extends Gradient {
//  override def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix):
//  (DoubleMatrix, Double) = {
//    val margin: Double = -1.0 * data.dot(weights)
//    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
//
//    val gradient = data.mul(gradientMultiplier)
//    val loss =
//      if (label > 0) {
//        math.log(1 + math.exp(margin))
//      } else {
//        math.log(1 + math.exp(margin)) - margin
//      }
//
//    (gradient, loss)
//  }
//
//  def computeMultiLogisticGradientAndHessian(data: DoubleMatrix, label: Double, weights: DoubleMatrix,
//      gradient: DoubleMatrix, hessian: DoubleMatrix) = {
//
//    val classes = (weights.rows / data.columns) + 1
//
//    def alpha(i: Double): Double = {
//      if (math.round(i) == 0) 1.0 else 0.0
//    }
//
//    def delta(i: Double, j: Double): Double = {
//      if (math.round(i) == math.round(j)) 1.0 else 0.0
//    }
//
//    var denominator: Double = 1.0
//    val numerators: Array[Double] = Array.ofDim[Double](classes - 1)
//
//    for (i <- 0 until classes - 1) {
//      var acc = weights.get(i, 0)
//      for (j <- 0 until x.length) {
//        acc += x(j) * w(i)(j)
//      }
//      numerators(i) = math.exp(acc)
//      denominator += numerators(i)
//    }
//
//    // gradient has dim of (classes-1) * (x.length+1)
//    // hessian has dim of (dim of grad)^2
//    for (i <- 0 until (classes - 1) * (x.length + 1)) {
//      val m1: Int = i % (x.length + 1) // m0 is intercept
//      val l1: Int = (i - m1) / (x.length + 1) // l + 1 is class
//      if (m1 == 0) {
//        gradient(i) += (1 - alpha(y)) * delta(y, l1 + 1) - numerators(l1) / denominator
//      } else {
//        gradient(i) += ((1 - alpha(y)) * delta(y, l1 + 1) - numerators(l1) / denominator) * x(m1 - 1)
//      }
//      for (j <- 0 until (classes - 1) * (x.length + 1)) {
//        val m2: Int = j % (x.length + 1)
//        val l2: Int = (j - m2) / (x.length + 1)
//        val temp: Double = {
//          if (m1 == 0 && m2 == 0)
//            1
//          else if (m1 == 0)
//            x(m2 - 1)
//          else if (m2 == 0)
//            x(m1 - 1)
//          else
//            x(m1 - 1) * x(m2 - 1)
//        }
//        if (l1 == l2) {
//          hessian(i)(j) -= temp * (numerators(l1) * (denominator - numerators(l1))) / (denominator * denominator)
//        } else {
//          hessian(i)(j) += temp * numerators(l1) * numerators(l2) / (denominator * denominator)
//        }
//      }
//    }
//  }
//}
//
//
////  def computeGradientAndHessian(y: Double, x: Array[Double], w: Array[Array[Double]], b: Array[Double],
////
////                                gradient: Array[Double], hessian: Array[Array[Double]]): Unit = {
////    val classes = b.length + 1
////
////    def alpha(i: Double): Double = {
////      if (math.round(i) == 0) 1.0 else 0.0
////    }
////
////    def delta(i: Double, j: Double): Double = {
////      if (math.round(i) == math.round(j)) 1.0 else 0.0
////    }
////
////    var denominator: Double = 1.0
////    val numerators: Array[Double] = Array.ofDim[Double](b.length)
////
////    for (i <- 0 until b.length) {
////      var acc = b(i)
////      for (j <- 0 until x.length) {
////        acc += x(j) * w(i)(j)
////      }
////      numerators(i) = math.exp(acc)
////      denominator += numerators(i)
////    }
////
////    // gradient has dim of (classes-1) * (x.length+1)
////    // hessian has dim of (dim of grad)^2
////    for (i <- 0 until (classes - 1) * (x.length + 1)) {
////      val m1: Int = i % (x.length + 1) // m0 is intercept
////      val l1: Int = (i - m1) / (x.length + 1) // l + 1 is class
////      if (m1 == 0) {
////        gradient(i) += (1 - alpha(y)) * delta(y, l1 + 1) - numerators(l1) / denominator
////      } else {
////        gradient(i) += ((1 - alpha(y)) * delta(y, l1 + 1) - numerators(l1) / denominator) * x(m1 - 1)
////      }
////      for (j <- 0 until (classes - 1) * (x.length + 1)) {
////        val m2: Int = j % (x.length + 1)
////        val l2: Int = (j - m2) / (x.length + 1)
////        val temp: Double = {
////          if (m1 == 0 && m2 == 0)
////            1
////          else if (m1 == 0)
////            x(m2 - 1)
////          else if (m2 == 0)
////            x(m1 - 1)
////          else
////            x(m1 - 1) * x(m2 - 1)
////        }
////        if (l1 == l2) {
////          hessian(i)(j) -= temp * (numerators(l1) * (denominator - numerators(l1))) / (denominator * denominator)
////        } else {
////          hessian(i)(j) += temp * numerators(l1) * numerators(l2) / (denominator * denominator)
////        }
////      }
////    }
////  }
////}
//
///**
//* Compute hessian and loss for a multinomial logistic loss function.
//*/
//object MultiLogisticHessian extends Hessian {
//  override def compute(data: DoubleMatrix, label: Double, weights: DoubleMatrix):
//  (DoubleMatrix, Double) = {
//    val margin: Double = -1.0 * data.dot(weights)
//    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
//
//    val gradient = data.mul(gradientMultiplier)
//    val loss =
//      if (label > 0) {
//        math.log(1 + math.exp(margin))
//      } else {
//        math.log(1 + math.exp(margin)) - margin
//      }
//
//    (gradient, loss)
//  }
//}
//
///**
//* Train a classification model for Multinomial Logistic Regression using Newton method.
//* NOTE: Labels used in Multinomial Logistic Regression should be {0, 1, 2, ..., N}
//* for N classes classification
//* @param classes Number of classes in the dependant variable
//* @param maxNumIterations Maximum number of iterations of Newton method to run.
//* @param regParam L2 regularization parameter
//* @param tolerance
//*/
//class MultiLogisticRegressionWithNewton private(
//                                                 val classes: Int,
//                                                 var maxNumIterations: Int,
//                                                 var regParam: Double,
//                                                 var tolerance: Double
//                                                 )
//  extends GeneralizedLinearAlgorithm[MultiLogisticRegressionModel]
//  with Serializable {
//
//  val gradient = MultiLogisticGradient
//  val hessian =  MultiLogisticHessian
//  val updater = new SimpleUpdater()
//    override val optimizer = new GradientDescent(gradient, updater)
//      .setStepSize(stepSize)
//      .setNumIterations(numIterations)
//      .setRegParam(regParam)
//      .setMiniBatchFraction(miniBatchFraction)
//    override val validators = List(DataValidators.multiClassificationLabels(classes))
//
//
//  /**
//   * Construct a LogisticRegression object with default parameters
//   */
//  def this(classes: Int) = this(classes, 100, 1e-4, 0.0)
//
//  override def createModel(weights: Array[Double], intercepts: Double) = {
//    new MultiLogisticRegressionModel(weights, classes)
//
//  }
//}
//
///**
//* Train a classification model for Multinomial Logistic Regression using Stochastic Gradient Descent.
//* NOTE: Labels used in Logistic Regression should be {0, 1}
//*/
//class MultiLogisticRegressionWithSGD private (
//                                               val classes: Int,
//                                               var stepSize: Double,
//                                               var numIterations: Int,
//                                               var regParam: Double,
//                                               var miniBatchFraction: Double)
//  extends GeneralizedLinearAlgorithm[MultiLogisticRegressionModel]
//  with Serializable {
//
//  val gradient = MultiLogisticGradient
//  val updater = new SimpleUpdater()
//  override val optimizer = new GradientDescent(gradient, updater)
//    .setStepSize(stepSize)
//    .setNumIterations(numIterations)
//    .setRegParam(regParam)
//    .setMiniBatchFraction(miniBatchFraction)
//  override val validators = List(DataValidators.multiClassificationLabels(classes))
//
//  /**
//   * Construct a LogisticRegression object with default parameters
//   */
//  def this() = this(1.0, 100, 0.0, 1.0)
//
//  def createModel(weights: Array[Double], intercept: Double) = {
//    new MultiLogisticRegressionModel(weights, classes)
//  }
//}
//
///**
//* Top-level methods for calling Logistic Regression.
//* NOTE: Labels used in Logistic Regression should be {0, 1}
//*/
//object MultiLogisticRegressionWithSGD {
//  // NOTE(shivaram): We use multiple train methods instead of default arguments to support
//  // Java programs.
//
//  /**
//   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
//   * number of iterations of gradient descent using the specified step size. Each iteration uses
//   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
//   * gradient descent are initialized using the initial weights provided.
//   * NOTE: Labels used in Logistic Regression should be {0, 1}
//   *
//   * @param input RDD of (label, array of features) pairs.
//   * @param numIterations Number of iterations of gradient descent to run.
//   * @param stepSize Step size to be used for each iteration of gradient descent.
//   * @param miniBatchFraction Fraction of data to be used per iteration.
//   * @param initialWeights Initial set of weights to be used. Array should be equal in size to
//   *        the number of features in the data.
//   */
//  def train(
//             input: RDD[LabeledPoint],
//             classes: Int,
//             numIterations: Int,
//             stepSize: Double,
//             miniBatchFraction: Double,
//             initialWeights: Array[Double])
//  : LogisticRegressionModel =
//  {
//    new MultiLogisticRegressionWithSGD(classes, stepSize, numIterations, 0.0, miniBatchFraction).run(
//      input, initialWeights)
//  }
//
//  /**
//   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
//   * number of iterations of gradient descent using the specified step size. Each iteration uses
//   * `miniBatchFraction` fraction of the data to calculate the gradient.
//   * NOTE: Labels used in Logistic Regression should be {0, 1}
//   *
//   * @param input RDD of (label, array of features) pairs.
//   * @param numIterations Number of iterations of gradient descent to run.
//   * @param stepSize Step size to be used for each iteration of gradient descent.
//
//   * @param miniBatchFraction Fraction of data to be used per iteration.
//   */
//  def train(
//             input: RDD[LabeledPoint],
//             numIterations: Int,
//             stepSize: Double,
//             miniBatchFraction: Double)
//  : LogisticRegressionModel =
//  {
//    new MultiLogisticRegressionWithSGD(stepSize, numIterations, 0.0, miniBatchFraction).run(
//      input)
//  }
//
//  /**
//   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
//   * number of iterations of gradient descent using the specified step size. We use the entire data
//   * set to update the gradient in each iteration.
//   * NOTE: Labels used in Logistic Regression should be {0, 1}
//   *
//   * @param input RDD of (label, array of features) pairs.
//   * @param stepSize Step size to be used for each iteration of Gradient Descent.
//
//   * @param numIterations Number of iterations of gradient descent to run.
//   * @return a LogisticRegressionModel which has the weights and offset from training.
//   */
//  def train(
//             input: RDD[LabeledPoint],
//             numIterations: Int,
//             stepSize: Double)
//  : LogisticRegressionModel =
//  {
//    train(input, numIterations, stepSize, 1.0)
//  }
//
//  /**
//   * Train a logistic regression model given an RDD of (label, features) pairs. We run a fixed
//   * number of iterations of gradient descent using a step size of 1.0. We use the entire data set
//   * to update the gradient in each iteration.
//   * NOTE: Labels used in Logistic Regression should be {0, 1}
//   *
//   * @param input RDD of (label, array of features) pairs.
//   * @param numIterations Number of iterations of gradient descent to run.
//   * @return a LogisticRegressionModel which has the weights and offset from training.
//   */
//  def train(
//             input: RDD[LabeledPoint],
//             numIterations: Int)
//  : LogisticRegressionModel =
//  {
//    train(input, numIterations, 1.0, 1.0)
//  }
//
//  def main(args: Array[String]) {
//    if (args.length != 4) {
//      println("Usage: LogisticRegression <master> <input_dir> <step_size> " +
//        "<niters>")
//      System.exit(1)
//    }
//    val sc = new SparkContext(args(0), "LogisticRegression")
//    val data = MLUtils.loadLabeledData(sc, args(1))
//    val model = LogisticRegressionWithSGD.train(data, args(3).toInt, args(2).toDouble)
//    println("Weights: " + model.weights.mkString("[", ", ", "]"))
//    println("Intercept: " + model.intercept)
//
//    sc.stop()
//  }
//}
