package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.matchers.should.Matchers._

import scala.util.{Failure, Success, Try}

class WEDTTests extends AnyFlatSpec with Matchers with Configuration {

  import sqlContext.implicits._
   "Text classifier" should "split text data from a file into single e-mails" in {

    val textClassifier = new OneVsRest().setClassifier(new LogisticRegression())
    val rdd = WEDT.prepareRdd("resources/tests/*")
    val result = rdd.collect
    val counts = result
      .groupBy(e => e.firstLevelLabelValue)
    assert(counts.get(0.0).get.length == 40)
    assert(counts.get(1.0).get.length == 20)

  }

  "Each row" should "be correctly labeled according to path" in {

    val rdd = WEDT.prepareRdd("resources/tests/*")
    rdd.collect
      .foreach(e => {
        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
    })
    WEDT.firstLevelLabelsMapping.size should be (2)
    WEDT.secondLevelLabelsMapping.size should be (3)
    WEDT.firstLevelLabelsMapping should contain ("alt" -> 1.0)
    WEDT.firstLevelLabelsMapping should contain ("comp" -> 0.0)
    WEDT.secondLevelLabelsMapping should contain ("atheism.txt" -> 0.0)
    WEDT.secondLevelLabelsMapping should contain ("graphics.txt" -> 1.0)
    WEDT.secondLevelLabelsMapping should contain ("os.ms-windows.misc.txt" -> 2.0)
  }
}
