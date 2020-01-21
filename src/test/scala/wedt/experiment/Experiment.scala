package wedt.experiment

import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import wedt._

class Experiment extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  val DataProvider =  new DataProvider("resources/20-newsgroups/", 0.07, 0.03)

  var metrics1: MulticlassMetrics = null
  var metrics2: MulticlassMetrics = null
  import sqlContext.implicits._

  "NaiveBayes single classifier, lambda=0.8" should "handle all classes" in {

    log.info(s"Starting Bayes single experiment")
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val slc = new SingleLayerClassifier(
      new NaiveBayes().setSmoothing(0.8),
      "bayes-single"
    )
    val pipeline = new TextPipeline(slc)
    log.info("starting fit for bayes-single")
    val trainedModel = pipeline.fit(DataProvider.trainDf)
    log.info("completed fit for bayes-single")
    val validationResult = trainedModel.transform(DataProvider.validateDf)
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
    ReadWriteToFileUtils.saveModel(trainedModel, "experiment/bayes-single.obj")

    metrics1 = new MulticlassMetrics(validationResult.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
    log.info(s"Confussion matrix (Bayes single):")
    println(metrics1.confusionMatrix)
    log.info(s"Accuracy  = $accuracy")
    log.info(s"Precision = $precision")
//    assert(accuracy == metrics1.accuracy)
//    assert(precision == metrics1.precision)
  }

  "NaiveBayes multi classifier, lambda=0.8" should "handle all classes" in {

    log.info(s"Starting Bayes multi experiment")
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
    val mlc = new MultilayerClassifier(
      new NaiveBayes().setSmoothing(0.8),
      (for {i <- 1 to 20} yield new NaiveBayes().setSmoothing(0.8)).toList,
      "bayes-multi"
    )
    log.info(s"size of DataProvider.trainDf: ${DataProvider.trainDf.count}")
    log.info(s"size of DataProvider.validateDf: ${DataProvider.validateDf.count}")
    log.info("starting fit for bayes-multi")
    val trainedModel = new TextPipeline(mlc).fit(DataProvider.trainDf)
    log.info("completed fit for bayes-multi")
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
    ReadWriteToFileUtils.saveModel(trainedModel, "experiment/bayes-multi.obj")

    metrics2 = new MulticlassMetrics(validationResult.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
    log.info(s"Confussion matrix (Bayes multi):")
    println(metrics2.confusionMatrix.toString(20, 1))
    log.info(s"Accuracy  = $accuracy")
    log.info(s"Precision = $precision")
//    assert(accuracy == metrics2.accuracy)
//    assert(precision == metrics2.precision)
  }

  "MulticlassMetrics" should "be saved" in {

    ReadWriteToFileUtils.saveModel(
        metrics1.confusionMatrix, "experiment/metrics1.obj")
    ReadWriteToFileUtils.saveModel(
        metrics2.confusionMatrix, "experiment/metrics2.obj")
  }
}
