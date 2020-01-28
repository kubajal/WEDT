package wedt.experiment

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizerModel, IndexToString}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import wedt._

class Experiment extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")
  import sparkSession.implicits._

  private val dataProvider =  new DataProvider("resources/20-newsgroups")
  private val df = dataProvider.prepareRddPerClass(1000)
    .toDF("firstLevelLabel", "secondLevelLabel", "features_0")
  private val Array(trainDf, validateDf) = df.randomSplit(Array(0.7, 0.3))


  var metrics1: MulticlassMetrics = _
  var metrics2: MulticlassMetrics = _
  val classesVocab = 1000

  "NaiveBayes single classifier" should "handle all classes" in {

    logSpark(s"Starting Bayes single experiment")
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val slc = new SingleLayerClassifier(
      new NaiveBayes().setSmoothing(1.0),
      "bayes-single"
    )
    val pipeline = new TextPipeline(slc, classesVocab)
    logSpark("starting fit for bayes-single")
    val trainedModel = pipeline.fit(trainDf)

    val vocabulary = trainedModel.stages.takeRight(3).head.asInstanceOf[CountVectorizerModel].vocabulary.toList
    logSpark(s"single: vocabulary for first level: $vocabulary")

    val indexer = slc.indexer
    val reverseIndexer = new IndexToString()
        .setLabels(indexer.labels)
        .setInputCol("label_0")
        .setOutputCol("label")

    logSpark("single: completed fit for bayes-single")
    val validationResult1 = trainedModel.transform(validateDf)
    val validationResult = indexer.transform(validationResult1)

    val accuracy = accuracyEvaluator.evaluate(validationResult)
    val precision = precisionEvaluator.evaluate(validationResult)

    import sparkSession.implicits._

    metrics1 = new MulticlassMetrics(validationResult.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
    logSpark(s"single: confussion matrix:")
    val labels = reverseIndexer.transform(metrics1.labels.toList.toDF("label_0"))
        .select("label").collect.map(e => e(0).asInstanceOf[String])

    logSpark(s"single: labels in confussion matrix${labels.toList}")
    metrics1.confusionMatrix.rowIter.foreach(e => println(e))
    logSpark(s"Accuracy  = $accuracy")
    logSpark(s"Precision = $precision")
//    assert(accuracy == metrics1.accuracy)
//    assert(precision == metrics1.precision)
  }

  "NaiveBayes multi classifier, lambda=0.8" should "handle all classes" in {

    logSpark(s"multi: Starting Bayes multi experiment")
    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
    val mlc = new MultilayerClassifier(
      new NaiveBayes().setSmoothing(1),
      (for {i <- 1 to 20} yield new NaiveBayes().setSmoothing(1)).toList,
      s"bayes-multi",
      500
    )

    logSpark(s"multi: size of DataProvider.trainDf: ${trainDf.count}")
    logSpark(s"multi: size of DataProvider.validateDf: ${validateDf.count}")
    logSpark("multi: starting fit for bayes-multi")
    val trainedModel = new TextPipeline(mlc, classesVocab).fit(trainDf)
    val vocabulary = trainedModel.stages.takeRight(3).head.asInstanceOf[CountVectorizerModel].vocabulary.toList
    val mlcm = trainedModel.stages.last.asInstanceOf[MultilayerClassificationModel]

    val result = trainedModel.transform(validateDf)

    logSpark(s"multi: result count: ${result.count}")

    logSpark("multi: completed fit for bayes-multi")
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

    metrics1 = new MulticlassMetrics(validationResult1.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
    metrics2 = new MulticlassMetrics(validationResult2.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
    logSpark(s"multi: confussion matrix for first level:")
    metrics1.confusionMatrix.rowIter.foreach(e => println(e))
    logSpark(s"multi: confussion matrix for second level:")
    metrics2.confusionMatrix.rowIter.foreach(e => println(e))

    val accuracy1 = accuracyEvaluator.evaluate(validationResult1)
    val precision1 = precisionEvaluator.evaluate(validationResult1)
    val accuracy2 = accuracyEvaluator.evaluate(validationResult2)
    val precision2 = precisionEvaluator.evaluate(validationResult2)
    logSpark(s"multi: First Level Accuracy   = $accuracy1")
    logSpark(s"multi: First Level Precision  = $precision1")
    logSpark(s"multi: Second Level Accuracy  = $accuracy2")
    logSpark(s"multi: Second Level Precision = $precision2")

  }
}
