package wedt

import java.io.File
import java.net.URL
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

import scala.util.{Failure, Success, Try}

object WEDT extends Configuration {
  import sqlContext.implicits._

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
        logger.info("liczba wczytanych plikow: " + plainText.count())

        plainText
          .map(e => (e._1, e._2, e._3))
          .flatMap(e => e._3
//            .split("((^(Newsgroup|From):.*((.)*\\n){0,11}^(From|Subject|Organization|: ===.*).*))")
            .split("From:")
            .filter(e => e != "")
            .map(f => TaggedText(e._1, e._2, f)))
          .persist
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        logger.info(s"Could not load files from the path: $path")
        sparkContext.stop()
        throw e
    }
  }

  def main(args: Array[String]): Unit = {


//    var params: scala.collection.mutable.Map[String, String] = scala.collection.mutable.Map(
//      "path" -> "",
//      "train" -> "0.1",
//      "validate" -> "0.1")
//
//    val newParams = args.reduce((a,b) => a + " " + b)
//      .split("--")
//      .filter(_ != "")
//      .map(e => e.split(" "))
//      .map(e => (e(0), e(1)))
//      .toMap
//
//    newParams.foreach(e => params.update(e._1, e._2))
//
//    println("params:")
//    println(params.toList)
//
//    val dataProvider =  new DataProvider(
//      params("path"),
//      params("train").toDouble,
//      params("validate").toDouble)
//
//      val accuracyEvaluator1 = new MulticlassClassificationEvaluator()
//        .setMetricName("accuracy")
//      val precisionEvaluator1 = new MulticlassClassificationEvaluator()
//        .setMetricName("weightedPrecision")
//
//      val slc1 = new SingleLayerClassifier(
//        new OneVsRest().setClassifier(new LogisticRegression()),
//        "regression-single"
//      )
//      val trainedModel1 = new TextPipeline(slc1).fit(dataProvider.trainDf)
//      val validationResult1 = trainedModel1.transform(dataProvider.validateDf)
//        .withColumnRenamed("secondLevelLabel", "label")
//      val accuracy = accuracyEvaluator1.evaluate(validationResult1)
//      val precision = precisionEvaluator1.evaluate(validationResult1)
//    validationResult1.map(e => (
//        e.getAs[String]("features_0")
//          .take(100)
//          .replace("\n", "")
//          .replace("\r", ""),
//        e.getAs[Double]("prediction"),
//        e.getAs[Double]("label")))
//        .show(numRows = 100, truncate = false)
//      logger.info(s"Accuracy  = $accuracy")
//      logger.info(s"Precision = $precision")
////      ReadWriteToFileUtils.saveModel(trainedModel1, train)
//
//      val accuracyEvaluator2 = new MulticlassClassificationEvaluator()
//        .setMetricName("accuracy")
//      val precisionEvaluator2 = new MulticlassClassificationEvaluator()
//        .setMetricName("weightedPrecision")
//
//      val slc2 = new SingleLayerClassifier(
//        new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8)),
//        "bayes-single"
//      )
//      val trainedModel2 = new TextPipeline(slc2).fit(dataProvider.trainDf)
//      val validationResult2 = trainedModel2.transform(dataProvider.validateDf)
//        .withColumnRenamed("secondLevelLabel", "label")
//      val accuracy2 = accuracyEvaluator2.evaluate(validationResult2)
//      val precision2 = precisionEvaluator2.evaluate(validationResult2)
//      validationResult2.map(e => (
//        e.getAs[String]("features_0")
//          .take(100)
//          .replace("\n", "")
//          .replace("\r", ""),
//        e.getAs[Double]("prediction"),
//        e.getAs[Double]("label")))
//        .show(numRows = 100, truncate = false)
//      logger.info(s"Accuracy  = $accuracy2")
//      logger.info(s"Precision = $precision2")
////      ReadWriteToFileUtils.saveModel(trainedModel2)
//
//      val accuracyEvaluator3 = new MulticlassClassificationEvaluator()
//        .setMetricName("accuracy")
//      val precisionEvaluator3 = new MulticlassClassificationEvaluator()
//        .setMetricName("weightedPrecision")
//
//      val mlc3 = new MultilayerClassifier(
//        new OneVsRest().setClassifier(new LogisticRegression()),
//        (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new LogisticRegression())).toList,
//        "regression-multi"
//      )
//      val trainedModel3 = new TextPipeline(mlc3).fit(dataProvider.trainDf)
//      val validationResult3 = trainedModel3.transform(dataProvider.validateDf)
//      val accuracy3 = accuracyEvaluator3.evaluate(validationResult3)
//      val precision3 = precisionEvaluator3.evaluate(validationResult3)
//      validationResult3.map(e => (
//        e.getAs[String]("features_0")
//          .take(100)
//          .replace("\n", "")
//          .replace("\r", ""),
//        e.getAs[Double]("prediction"),
//        e.getAs[Double]("label")))
//        .show(numRows = 100, truncate = false)
//      logger.info(s"Accuracy  = $accuracy3")
//      logger.info(s"Precision = $precision3")
////      ReadWriteToFileUtils.saveModel(trainedModel3)
//
//      val accuracyEvaluator4 = new MulticlassClassificationEvaluator()
//        .setMetricName("accuracy")
//      val precisionEvaluator4 = new MulticlassClassificationEvaluator()
//        .setMetricName("weightedPrecision")
//      val mlc4 = new MultilayerClassifier(
//        new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8)),
//        (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new NaiveBayes().setSmoothing(0.8))).toList,
//        "bayes-multi"
//      )
//      val trainedModel4 = new TextPipeline(mlc4).fit(dataProvider.trainDf)
//      val validationResult4 = trainedModel4.transform(dataProvider.validateDf)
//      val accuracy4 = accuracyEvaluator4.evaluate(validationResult4)
//      val precision4 = precisionEvaluator4.evaluate(validationResult4)
//      validationResult4.map(e => (
//        e.getAs[String]("features_0")
//          .take(100)
//          .replace("\n", "")
//          .replace("\r", ""),
//        e.getAs[Double]("prediction"),
//        e.getAs[Double]("label")))
//        .show(numRows = 100, truncate = false)
//      logger.info(s"Accuracy  = $accuracy4")
//      logger.info(s"Precision = $precision4")
//      ReadWriteToFileUtils.saveModel(trainedModel4)
  }
}
