package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success}

object WEDT extends App {

  private val defaultPath = "resources/20-newsgroups/*"

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

  override def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc: SparkContext = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val textClassifier = new TextClassifier(sc, preparePipeline())
    val path = if(args.length == 0) "resources/20-newsgroups/*" else args.head
    textClassifier.learn(textClassifier.prepareRdd(path))
    textClassifier.test()
  }
}
