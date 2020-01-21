package wedt.multiLayer

import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import wedt._

class MultilayerLogisticRegressionTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  import sqlContext.implicits._

  val DataProvider =  new DataProvider("resources/20-newsgroups/*", 0.1, 0.1)

  "LogisticRegression classifier" should "handle four classes" in {

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val mlc = new MultilayerClassifier(
      new OneVsRest().setClassifier(new LogisticRegression()),
      (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new LogisticRegression())).toList,
      "regression-multi"
    )
    val trainedModel = new TextPipeline(mlc).fit(DataProvider.trainDf)
    val validationResult = trainedModel.transform(DataProvider.validateDf)
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
//    ReadWriteToFileUtils.saveModel(trainedModel)
  }

//  "Classifier" should "handle all classes" in {
//
//    val accuracyEvaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("accuracy")
//    val precisionEvaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("weightedPrecision")
//
//    val rdd = WEDT.prepareRdd("resources/20-newsgroups/*")
//    rdd.collect
//      .foreach(e => {
//        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
//        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
//      })
//
//    val Array(train, validate, rest) = rdd
//      .toDF()
//      .withColumnRenamed("text", "features_0")
//      .randomSplit(Array(0.1, 0.1, 0.8))
//
//    val mlc = new MultilayerClassifier(
//      new OneVsRest().setClassifier(new LogisticRegression()),
//      (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new LogisticRegression())).toList,
//      "mlc"
//    )
//
//    val trainedModel = new TextPipeline(mlc).fit(train)
//    val validationResult = trainedModel.transform(validate)
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
