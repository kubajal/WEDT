package wedt

import java.util.Calendar

import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}
import wedt.DataProvider.sqlContext
import wedt.WEDT.firstLevelLabelsMapping

import scala.util.{Failure, Success, Try}

object WEDT extends Configuration {
  import sqlContext.implicits._

  var firstLevelLabelsMapping: Map[String, Double] = _
  var secondLevelLabelsMapping: Map[String, Double] = _

  def prepareRdd(path: String): RDD[TaggedText] = {

    val plainTextTry = Try(sparkContext.wholeTextFiles(path))
    plainTextTry match {
      case Success(textData) =>

        // wyciaganie sciezki
        val plainText1 = textData
          .map(e => (e._1.split("/").takeRight(1).reduce((a,b) => a+"/"+b), e._2))
        val plainText2 = plainText1
          .map(e => (e._1.split("\\."), e._2))
        val plainText = plainText2
          .map(e => (e._1.head, e._1.takeRight(e._1.length-1).reduce((a,b) => a+"."+b), e._2))

        //todo: zrobic porzadne logowanie
        log.info("liczba wczytanych plikow: " + plainText.count())

        firstLevelLabelsMapping =  plainText
          .map(e => e._1)
          .distinct()
          .zipWithIndex
          .map(e => (e._1, e._2.toDouble))
          .collect()
          .toList
          .toMap

        secondLevelLabelsMapping =  plainText
          .map(e => e._2)
          .zipWithIndex
          .map(e => (e._1, e._2.toDouble))
          .collect()
          .toList
          .toMap

        plainText
          .map(e => (e._1, firstLevelLabelsMapping(e._1), e._2, secondLevelLabelsMapping(e._2), e._3))
          .flatMap(e => e._5
            .split("From:")
            .filter(e => e != "")
            .map(f => TaggedText(e._1, e._2, e._3, e._4, f)))
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        log.info(s"Could not load files from the path: $path")
        sparkContext.stop()
        throw e
    }
  }

  def main(args: Array[String]): Unit = {

      val accuracyEvaluator1 = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")
      val precisionEvaluator1 = new MulticlassClassificationEvaluator()
        .setMetricName("weightedPrecision")

      val slc1 = new SingleLayerClassifier(
        new OneVsRest().setClassifier(new LogisticRegression()),
        "regression-single"
      )
      val trainedModel1 = new TextPipeline(slc1).fit(DataProvider.train)
      val validationResult1 = trainedModel1.transform(DataProvider.validate)
        .withColumnRenamed("secondLevelLabelValue", "label")
      val accuracy = accuracyEvaluator1.evaluate(validationResult1)
      val precision = precisionEvaluator1.evaluate(validationResult1)
    validationResult1.map(e => (
        e.getAs[String]("features_0")
          .take(100)
          .replace("\n", "")
          .replace("\r", ""),
        e.getAs[Double]("prediction"),
        e.getAs[Double]("label")))
        .show(numRows = 100, truncate = false)
      log.info(s"Accuracy  = $accuracy")
      log.info(s"Precision = $precision")
      ReadWriteToFileUtils.saveModel(trainedModel1)

      val accuracyEvaluator2 = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")
      val precisionEvaluator2 = new MulticlassClassificationEvaluator()
        .setMetricName("weightedPrecision")

      val slc2 = new SingleLayerClassifier(
        new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8)),
        "bayes-single"
      )
      val trainedModel2 = new TextPipeline(slc2).fit(DataProvider.train)
      val validationResult2 = trainedModel2.transform(DataProvider.validate)
        .withColumnRenamed("secondLevelLabelValue", "label")
      val accuracy2 = accuracyEvaluator2.evaluate(validationResult2)
      val precision2 = precisionEvaluator2.evaluate(validationResult2)
      validationResult2.map(e => (
        e.getAs[String]("features_0")
          .take(100)
          .replace("\n", "")
          .replace("\r", ""),
        e.getAs[Double]("prediction"),
        e.getAs[Double]("label")))
        .show(numRows = 100, truncate = false)
      log.info(s"Accuracy  = $accuracy2")
      log.info(s"Precision = $precision2")
      ReadWriteToFileUtils.saveModel(trainedModel2)

      val accuracyEvaluator3 = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")
      val precisionEvaluator3 = new MulticlassClassificationEvaluator()
        .setMetricName("weightedPrecision")

      val mlc3 = new MultilayerClassifier(
        new OneVsRest().setClassifier(new LogisticRegression()),
        (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new LogisticRegression())).toList,
        "regression-multi"
      )
      val trainedModel3 = new TextPipeline(mlc3).fit(DataProvider.train)
      val validationResult3 = trainedModel3.transform(DataProvider.validate)
      val accuracy3 = accuracyEvaluator3.evaluate(validationResult3)
      val precision3 = precisionEvaluator3.evaluate(validationResult3)
      validationResult3.map(e => (
        e.getAs[String]("features_0")
          .take(100)
          .replace("\n", "")
          .replace("\r", ""),
        e.getAs[Double]("prediction"),
        e.getAs[Double]("label")))
        .show(numRows = 100, truncate = false)
      log.info(s"Accuracy  = $accuracy3")
      log.info(s"Precision = $precision3")
      ReadWriteToFileUtils.saveModel(trainedModel3)

      val accuracyEvaluator4 = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")
      val precisionEvaluator4 = new MulticlassClassificationEvaluator()
        .setMetricName("weightedPrecision")
      val mlc4 = new MultilayerClassifier(
        new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8)),
        (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8))).toList,
        "bayes-multi"
      )
      val trainedModel4 = new TextPipeline(mlc4).fit(DataProvider.train)
      val validationResult4 = trainedModel4.transform(DataProvider.validate)
      val accuracy4 = accuracyEvaluator4.evaluate(validationResult4)
      val precision4 = precisionEvaluator4.evaluate(validationResult4)
      validationResult4.map(e => (
        e.getAs[String]("features_0")
          .take(100)
          .replace("\n", "")
          .replace("\r", ""),
        e.getAs[Double]("prediction"),
        e.getAs[Double]("label")))
        .show(numRows = 100, truncate = false)
      log.info(s"Accuracy  = $accuracy4")
      log.info(s"Precision = $precision4")
      ReadWriteToFileUtils.saveModel(trainedModel4)
  }
}
