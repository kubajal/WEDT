package wedt

import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

object WEDT extends Configuration {

  import sparkSession.implicits._

  private val dataProvider =  new DataProvider("classifier/resources/20-newsgroups")

  val Array(trainDf, validateDf) = dataProvider.prepareRddPerClass(100).persist
    .toDF("firstLevelLabel", "secondLevelLabel", "features_0")
    .randomSplit(Array(0.7, 0.3))

  var metrics1: MulticlassMetrics = _
  var metrics2: MulticlassMetrics = _
  val recallEval = new MulticlassClassificationEvaluator()
    .setMetricName("weightedRecall")
  val precisionEvaluator = new MulticlassClassificationEvaluator()
    .setMetricName("weightedPrecision")


  def main(args: Array[String]): Unit = {

    for {
      firstLevelVocabSize <- 2000 to 2000 by 100
    } yield test(firstLevelVocabSize, List(2000), trainDf, validateDf)
//    logSpark(s"testing with 1000 per subclass")
//    for {
//      firstLevelVocabSize <- List(100, 1000, 10000)
//    } yield test(firstLevelVocabSize, List(100, 1000, 10000), trainDf2, validateDf2)

  }

  def test(firstLevelVocabSize: Int, secondLevelVocabSizes: List[Int], trainDf: DataFrame, validateDf: DataFrame): Unit = {

    logSpark(s"NEW TEST | firstLevelVocabSize: $firstLevelVocabSize, secondLevelVocabSize: $secondLevelVocabSizes")
    logSpark(s"firstLevelVocabSize,secondLevelVocabSize,type,firstLevelWeightedPrecision,firstLevelWeightedRecall,secondLevelWeightedPrecision,secondLevelWeightedRecall")

//    logSpark(s"Starting Bayes single experiment")
    val slc = new SingleLayerClassifier(
      new NaiveBayes().setSmoothing(1.0),
      "bayes-single"
    )
    val pipeline = new TextPipeline(slc, firstLevelVocabSize)
//    logSpark("starting fit for bayes-single")
    val trainedModel1 = pipeline.fit(trainDf)

    val vocabulary = trainedModel1.stages.takeRight(3).head.asInstanceOf[CountVectorizerModel].vocabulary.toList
//    logSpark(s"single: vocabulary for first level: $vocabulary")

    val indexer = slc.indexer
    val reverseIndexer = new IndexToString()
      .setLabels(indexer.labels)
      .setInputCol("label_0")
      .setOutputCol("label")

//    logSpark("single: completed fit for bayes-single")
    val validationResult11 = trainedModel1.transform(validateDf)
    val validationResult1 = indexer.transform(validationResult11)
//
//    val recall = recallEval.evaluate(validationResult1)
//    val precision = precisionEvaluator.evaluate(validationResult1)
    import sparkSession.implicits._

    metrics1 = new MulticlassMetrics(validationResult1.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
//    logSpark(s"single: confussion matrix:")
//    val labels = reverseIndexer.transform(metrics1.labels.toList.toDF("label_0"))
//      .select("label").collect.map(e => e(0).asInstanceOf[String])

    val labels = metrics1.labels.map(e => indexer.labels(e.toInt)).toList
    logSpark(s"$firstLevelVocabSize,NA,single-bayes,${metrics1.weightedPrecision},${metrics1.weightedRecall},NA,NA")
    logSpark(s"single: labels in confussion matrix${labels}")
    metrics1.confusionMatrix.rowIter.foreach(e => println(e.toArray.toList.map(f => f.toInt)))
//    logSpark(s"Accuracy  = $recall")
//    logSpark(s"Precision = $precision")

    for{
      secondLevelVocabSize <- secondLevelVocabSizes
    } yield {
      logSpark(s"multi: Starting Bayes multi experiment")
      val mlc = new MultilayerClassifier(
        new NaiveBayes().setSmoothing(1),
        (for {i <- 1 to 7} yield new NaiveBayes().setSmoothing(1)).toList,
        s"bayes-multi",
        secondLevelVocabSize
      )

//      logSpark(s"multi: size of DataProvider.trainDf: ${trainDf.count}")
//      logSpark(s"multi: size of DataProvider.validateDf: ${validateDf.count}")
//      logSpark("multi: starting fit for bayes-multi")
      val trainedModel = new TextPipeline(mlc, firstLevelVocabSize).fit(trainDf)
      val mlcm = trainedModel.stages.last.asInstanceOf[MultilayerClassificationModel]

      val result = trainedModel.transform(validateDf)

//      logSpark(s"multi: result count: ${result.count}")

//      logSpark("multi: completed fit for bayes-multi")
      val validationResult12 = mlcm.firstLevelIndexer.transform(result
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

      metrics1 = new MulticlassMetrics(validationResult12.rdd
        .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
      metrics2 = new MulticlassMetrics(validationResult2.rdd
        .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))
//      logSpark(s"multi: confussion matrix for first level:")
//      metrics1.confusionMatrix.rowIter.foreach(e => println(e))
//      logSpark(s"multi: confussion matrix for second level:")
//      metrics2.confusionMatrix.rowIter.foreach(e => println(e))


      val labels1 = metrics1.labels.map(e => mlcm.firstLevelIndexer.labels(e.toInt)).toList
      val labels2 = metrics2.labels.map(e => mlcm.globalIndexer.labels(e.toInt)).toList
      logSpark(s"$firstLevelVocabSize,$secondLevelVocabSize,multi-bayes,${metrics1.weightedPrecision},${metrics1.weightedRecall},,${metrics2.weightedPrecision},${metrics2.weightedRecall}")
      logSpark(s"multi: labels on first level confusion matrix${labels1}")
      metrics1.confusionMatrix.rowIter.foreach(e => println(e.toArray.toList.map(f => f.toInt)))
      logSpark(s"multi: labels on second level confusion matrix${labels2}")
      metrics2.confusionMatrix.rowIter.foreach(e => println(e.toArray.toList.map(f => f.toInt)))

//      val accuracy1 = recallEval.evaluate(validationResult12)
//      val precision1 = precisionEvaluator.evaluate(validationResult12)
//      val accuracy2 = recallEval.evaluate(validationResult2)
//      val precision2 = precisionEvaluator.evaluate(validationResult2)
//      logSpark(s"multi: First Level Accuracy   = $accuracy1")
//      logSpark(s"multi: First Level Precision  = $precision1")
//      logSpark(s"multi: Second Level Accuracy  = $accuracy2")
//      logSpark(s"multi: Second Level Precision = $precision2")
    }
  }
}
