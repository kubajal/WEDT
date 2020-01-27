package wedt.singleLayer

import org.apache.spark.ml.classification.{NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import wedt._

class SingleLayerBayesTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  import sqlContext.implicits._

  val dataProvider =  new DataProvider("resources/20-newsgroups")
  val df = dataProvider.prepareRdd1(100)
    .toDF("firstLevelLabel", "secondLevelLabel", "features_0")
  val Array(trainDf, validateDf) = df.randomSplit(Array(0.7, 0.3))

  "NaiveBayes classifier, lambda=0.8" should "handle all classes" in {

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val slc = new SingleLayerClassifier(
      new NaiveBayes().setSmoothing(0.8),
      "bayes-single"
    )
    val pipeline = new TextPipeline(slc, 300)
    val trainedModel = pipeline.fit(trainDf)
    val validationResult = trainedModel.transform(validateDf)
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
    logger.info(s"Accuracy  = $accuracy")
    logger.info(s"Precision = $precision")
    val metrics = new MulticlassMetrics(validationResult.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
    logger.info(s"Confussion matrix 1 (Bayes single):")
    println(metrics.confusionMatrix.toString())
    logger.info(s"Confussion matrix 2 (Bayes single):")
    println(metrics.confusionMatrix.toString(20, 20))
    val cm = metrics.confusionMatrix
    cm.rowIter.foreach(e => println(e))
//    ReadWriteToFileUtils.saveModel(trainedModel)
  }

//  "NaiveBayes classifier, lambda=0.8" should "handle four classes" in {
//
//    val accuracyEvaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("accuracy")
//
//    val precisionEvaluator = new MulticlassClassificationEvaluator()
//      .setMetricName("weightedPrecision")
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
