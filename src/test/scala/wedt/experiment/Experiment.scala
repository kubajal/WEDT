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
  import Implicits._

  val DataProvider =  new DataProvider("resources/20-newsgroups", 0.07, 0.03)

  var metrics1: MulticlassMetrics = _
  var metrics2: MulticlassMetrics = _
  import sqlContext.implicits._

  "NaiveBayes single classifier" should "handle all classes" in {

    logger.info(s"Starting Bayes single experiment")
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val slc = new SingleLayerClassifier(
      new NaiveBayes().setSmoothing(0.0),
      "bayes-single"
    )
    val pipeline = new TextPipeline(slc)
    logger.info("starting fit for bayes-single")
    val trainedModel = pipeline.fit(DataProvider.trainDf)

    val indexer = slc.indexer

    logger.info("completed fit for bayes-single")
    val validationResult1 = trainedModel.transform(DataProvider.validateDf)
    val validationResult = indexer.transform(validationResult1)

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
    logger.info(s"Confussion matrix (Bayes single):")
    println(metrics1.confusionMatrix)
    logger.info(s"Accuracy  = $accuracy")
    logger.info(s"Precision = $precision")
//    assert(accuracy == metrics1.accuracy)
//    assert(precision == metrics1.precision)
  }

  "NaiveBayes multi classifier, lambda=0.8" should "handle all classes" in {

    logger.info(s"Starting Bayes multi experiment")
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
    val mlc = new MultilayerClassifier(
      new NaiveBayes().setSmoothing(0.8),
      (for {i <- 1 to 20} yield new NaiveBayes().setSmoothing(0.8)).toList,
      "bayes-multi"
    )

    logger.info(s"size of DataProvider.trainDf: ${DataProvider.trainDf.count}")
    logger.info(s"size of DataProvider.validateDf: ${DataProvider.validateDf.count}")
    logger.info("starting fit for bayes-multi")
    val trainedModel = new TextPipeline(mlc).fit(DataProvider.trainDf)
    val mlcm = trainedModel.stages.last.asInstanceOf[MultilayerClassificationModel]

    val result = trainedModel.transform(DataProvider.validateDf)

    println(s"result count: ${result.count}")
    result.select("predicted1stLevelClass", "firstLevelLabel", "predicted2ndLevelClass", "secondLevelLabel")
      .show(numRows = 500, truncate = false)
    result.printSchema()

    logger.info("completed fit for bayes-multi")
    val validationResult1 = mlcm.firstLevelIndexer.transform(result
      .drop("prediction")
      .drop("label")
      .withColumnRenamed("1stLevelPrediction", "prediction"))
    mlcm.globalIndexer
      .setInputCol("secondLevelLabel")
      .setOutputCol("label")
    val validationResult2_tmp = mlcm.globalIndexer.transform(result
      .drop("prediction")
      .drop("label"))
    val validationResult2 = mlcm.globalIndexer
      .setInputCol("predicted2ndLevelClass")
      .setOutputCol("prediction")
      .transform(validationResult2_tmp)

    validationResult1.printSchema()
    validationResult1.show(false)

    val accuracy1 = accuracyEvaluator.evaluate(validationResult1)
    val precision1 = precisionEvaluator.evaluate(validationResult1)
    val accuracy2 = accuracyEvaluator.evaluate(validationResult2)
    val precision2 = precisionEvaluator.evaluate(validationResult2)
    logger.info(s"First Level Accuracy   = $accuracy1")
    logger.info(s"First Level Precision  = $precision1")
    logger.info(s"Second Level Accuracy  = $accuracy2")
    logger.info(s"Second Level Precision = $precision2")

//    result.map(e => (
//      e.getAs[String]("features_0")
//        .take(100)
//        .replace("\n", "")
//        .replace("\r", ""),
//      e.getAs[Double]("prediction"),
//      e.getAs[Double]("firstLevelLabel"),
//      e.getAs[Double]("secondLevelLabel")))
//      .show(numRows = 100, truncate = false)
//    val validationResult = trainedModel.transform(DataProvider.validateDf)
//    val accuracy = accuracyEvaluator.evaluate(validationResult)
//    val precision = precisionEvaluator.evaluate(validationResult)
//    ReadWriteToFileUtils.saveModel(trainedModel, "experiment/bayes-multi.obj")
//
//    metrics2 = new MulticlassMetrics(validationResult.rdd
//      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
//    logger.info(s"Confussion matrix (Bayes multi):")
//    println(metrics2.confusionMatrix.toString(20, 1))
//    assert(accuracy == metrics2.accuracy)
//    assert(precision == metrics2.precision)
  }

//  "MulticlassMetrics" should "be saved" in {
//
//    ReadWriteToFileUtils.saveModel(
//        metrics1.confusionMatrix, "experiment/metrics1.obj")
//    ReadWriteToFileUtils.saveModel(
//        metrics2.confusionMatrix, "experiment/metrics2.obj")
//  }
}
