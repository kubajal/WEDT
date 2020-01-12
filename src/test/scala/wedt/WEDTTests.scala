package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.{Failure, Success, Try}

class WEDTTests extends AnyFlatSpec with Matchers {

  val sqlContext: SQLContext = SparkSession.builder.getOrCreate().sqlContext
  val sc: SparkContext = SparkSession.builder.getOrCreate().sparkContext
  import sqlContext.implicits._

  def preparePipeline(): Pipeline = {

    val tokenizer = new Tokenizer()
      .setInputCol("features_0")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("features_1")
    val punctuationRemover = new PunctuationRemover("punctuationRemover")
      .setInputCol("features_2")
    val stemmer = new PorterStemmerWrapper("stemmer")
      .setInputCol("features_3")
    val tf = new HashingTF()
      .setInputCol("features_4")
    val idf = new IDF()
      .setInputCol("features_5")

    tokenizer.setOutputCol(stopWordsRemover.getInputCol)
    stopWordsRemover.setOutputCol(punctuationRemover.getInputCol)
    punctuationRemover.setOutputCol(stemmer.getInputCol)
    stemmer.setOutputCol(tf.getInputCol)
    tf.setOutputCol(idf.getInputCol)
    idf.setOutputCol("features")

    new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, punctuationRemover, stemmer, tf, idf))
  }

  "Text classifier" should "split text data from a file into single e-mails" in {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc: SparkContext = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val textClassifier = new OneVsRestClassifier(new NaiveBayes)
    val rdd = WEDT.prepareRdd("resources/tests/*")
    val result = rdd.collect
    val counts = result
        .groupBy(e => e._1)
    assert(counts.get(0.0).get.length == 20)
    assert(counts.get(1.0).get.length == 20)
  }
}
