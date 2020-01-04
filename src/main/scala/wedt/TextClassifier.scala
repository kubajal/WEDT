package wedt

import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

object TextClassifier {

  private val log = LogManager.getRootLogger
  private val defaultPath = "resources/20-newsgroups/*"

  def main(args: Array[String]) {

    //todo: zrobic porzadne logowanie
    println(s"Pass the path to the dataset as the first argument. Default is '$defaultPath'.")

    val path =
      if(args.head == "")
        defaultPath
      else
        args.head
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val plainTextTry = Try(sc.wholeTextFiles(path))
    plainTextTry match {
      case Success(plainText) =>

        //todo: zrobic porzadne logowanie
        println("liczba wczytanych plikow: " + plainText.count())

        val sqlContext = SparkSession.builder.getOrCreate().sqlContext
        import sqlContext.implicits._
        val df = plainText
          .map(e => e._2)
          .toDF()

        val tokenizer = new Tokenizer()
          .setInputCol("value")
          .setOutputCol("value1")
        val stopWordsRemover = new StopWordsRemover()
          .setInputCol("value1")
          .setOutputCol("value2")
        val punctuationRemover = new PunctuationRemover("punctuationRemover")
          .setInputCol("value2")
          .setOutputCol("value3")
        val stemmer = new PorterStemmerWrapper("stemmer")
          .setInputCol("value3")
          .setOutputCol("value4")
        val tf = new HashingTF()
          .setInputCol(tokenizer.getOutputCol)
          .setOutputCol("tf")
        val idf = new IDF()
          .setInputCol(tf.getOutputCol)
          .setOutputCol("tfidf")

        val pipeline = new Pipeline()
          .setStages(Array(tokenizer, stopWordsRemover, punctuationRemover, stemmer, tf, idf))

        val result = pipeline.fit(df)
          .transform(df)

        result
          .select("tfidf")
          .show(false)

        sc.stop()
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        println(s"Could not load files from the path: $path")
        sc.stop()
    }
  }

}