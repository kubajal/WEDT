package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.matchers.should.Matchers._
import wedt.WEDT.{log, sqlContext}

import scala.util.{Failure, Success, Try}

class WEDTTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")
  import sqlContext.implicits._

  import sqlContext.implicits._
   "Text classifier" should "split text data from a file into single e-mails" in {

    val textClassifier = new OneVsRest().setClassifier(new LogisticRegression())
    val rdd = WEDT.prepareRdd("resources/tests/*")
    val result = rdd.collect
    val counts = result
      .groupBy(e => e.firstLevelLabelValue)
    assert(counts.get(0.0).get.length == 10)
    assert(counts.get(1.0).get.length == 10)
   assert(counts.get(2.0).get.length == 10)
   assert(counts.get(3.0).get.length == 10)

  }

  "Each row" should "be correctly labeled according to path" in {

    val rdd = WEDT.prepareRdd("resources/tests/*")
    rdd.collect
      .foreach(e => {
        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
      })
    WEDT.firstLevelLabelsMapping.size should be (4)
    WEDT.secondLevelLabelsMapping.size should be (4)
    WEDT.firstLevelLabelsMapping should contain ("soc" -> 0.0)
    WEDT.firstLevelLabelsMapping should contain ("alt" -> 2.0)
    WEDT.firstLevelLabelsMapping should contain ("comp" -> 1.0)
    WEDT.firstLevelLabelsMapping should contain ("rec" -> 3.0)
    WEDT.secondLevelLabelsMapping should contain ("atheism.txt" -> 0.0)
    WEDT.secondLevelLabelsMapping should contain ("graphics.txt" -> 1.0)
    WEDT.secondLevelLabelsMapping should contain ("sport.baseball.txt" -> 2.0)
    WEDT.secondLevelLabelsMapping should contain ("religion.christian.txt" -> 3.0)
  }

  "Classifier" should "handle four classes" in {

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val rdd = WEDT.prepareRdd("resources/tests/*")
    rdd.collect
      .foreach(e => {
        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
      })

    val Array(train, validate) = rdd
      .toDF()
      .withColumnRenamed("text", "features_0")
      .randomSplit(Array(0.7, 0.3))
    val trainedModel = new TextPipeline().fit(train)
    val validationResult = trainedModel.transform(validate)
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
  }
}
