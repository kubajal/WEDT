package wedt

import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, OneVsRest}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

class TextClassifier(val sc: SparkContext) {

  private val defaultPath = "resources/20-newsgroups/*"

  def prepareDf(path: String): Try[DataFrame] = {

    val plainTextTry = Try(sc.wholeTextFiles(path))
    plainTextTry match {
      case Success(plainText) =>

        //todo: zrobic porzadne logowanie
        println("liczba wczytanych plikow: " + plainText.count())

        val sqlContext = SparkSession.builder.getOrCreate().sqlContext
        import sqlContext.implicits._

        Success(plainText
          .zipWithIndex
          .map(e => (e._1._1, e._1._2, e._2.toDouble))
          .flatMap(e => e._2.split("From:").filter(e => e != "").map(f => (e._3, f.take(100))))
          .toDF()
          .withColumnRenamed("_2", "features_0")
          .withColumnRenamed("_1", "label"))
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        println(s"Could not load files from the path: $path")
        sc.stop()
        Failure(e)
    }
  }

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

  def run(path: String): Unit = {

        prepareDf(path) match {
          case Success(df) =>

            val result = preparePipeline().fit(df)
            .transform(df)
            .select("label", "features")

            val classifier = new NaiveBayes()

            val oneVsRest = new OneVsRest().setClassifier(classifier)

            val model = oneVsRest.fit(result)

            sc.stop()
          case Failure(e) =>
            throw e
        }

    }
}