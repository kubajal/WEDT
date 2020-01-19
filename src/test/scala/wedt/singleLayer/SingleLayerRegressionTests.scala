package wedt.singleLayer

import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import wedt._
import wedt.experiment.DataProvider

class SingleLayerRegressionTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  import sqlContext.implicits._

  "LogisticRegression single classifier, lambda=0.8" should "handle all classes" in {

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val slc = new SingleLayerClassifier(
      new OneVsRest().setClassifier(new LogisticRegression()),
      "regression-single"
    )
    val trainedModel = new TextPipeline(slc).fit(DataProvider.train)
    val validationResult = trainedModel.transform(DataProvider.validate)
      .withColumnRenamed("secondLevelLabelValue", "label")
    val accuracy = accuracyEvaluator.evaluate(validationResult)
    val precision = precisionEvaluator.evaluate(validationResult)
    validationResult.map(e => (
      e.getAs[String]("features_0")
        .take(100)
        .replace("\n", "")
        .replace("\r", ""),
      e.getAs[Double]("prediction"),
      e.getAs[Double]("label")))
      .show(numRows = 100, truncate = false)
    log.info(s"Accuracy  = $accuracy")
    log.info(s"Precision = $precision")
    ReadWriteToFileUtils.saveModel(trainedModel)
  }

//  "NaiveBayes classifier, lambda=0.8" should "handle four classes" in {
//
//    val accuracyEvaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("accuracy")
//
//    val precisionEvaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("weightedPrecision")
//
//    val rdd = WEDT.prepareRdd("resources/tests/*")
//    rdd.collect
//      .foreach(e => {
//        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
//        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
//      })
//
//    val Array(train, validate) = rdd
//      .toDF()
//      .withColumnRenamed("text", "features_0")
//      .randomSplit(Array(0.7, 0.3))
//
//
//    val slc = new SingleLayerClassifier(
//      new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8)),
//      "bayes"
//    )
//    val trainedModel = new TextPipeline(slc).fit(train)
//    val validationResult = trainedModel.transform(validate)
//      .withColumnRenamed("secondLevelLabelValue", "label")
//    val accuracy = accuracyEvaluator.evaluate(validationResult)
//    val precision = precisionEvaluator.evaluate(validationResult)
//    validationResult.map(e => (
//      e.getAs[String]("features_0")
//        .take(100)
//        .replace("\n", "")
//        .replace("\r", ""),
//      e.getAs[Double]("prediction"),
//      e.getAs[Double]("label")))
//      .show(numRows = 100, truncate = false)
//    log.info(s"Accuracy  = $accuracy")
//    log.info(s"Precision = $precision")
//  }
}
