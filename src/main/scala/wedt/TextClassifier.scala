package wedt

import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

class TextClassifier(val sc: SparkContext) {

  private val defaultPath = "resources/20-newsgroups/*"

  def prepareRdd(path: String): Try[RDD[(Double, String)]] = {

    val plainTextTry = Try(sc.wholeTextFiles(path))
    plainTextTry match {
      case Success(plainText) =>

        //todo: zrobic porzadne logowanie
        println("liczba wczytanych plikow: " + plainText.count())

        Success(plainText
          .zipWithIndex
          .map(e => (e._1._1, e._1._2, e._2.toDouble))
          .flatMap(e => e._2
            .split("From:")
            .filter(e => e != "")
            .map(f => (e._3, f.take(100)))))
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

        prepareRdd(path) match {
          case Success(rdd) =>

            val sqlContext = SparkSession.builder.getOrCreate().sqlContext
            import sqlContext.implicits._

            val df = rdd.toDF()
              .withColumnRenamed("_1", "label")
              .withColumnRenamed("_2", "features_0")

            val Array(train, test) = preparePipeline().fit(df)
              .transform(df)
              .select("label", "features")
              .randomSplit(Array(0.7, 0.3))

            println("random split -- train: " + train.count + ", test: " + test.count)

            val oneVsRest = new OneVsRest().setClassifier(new NaiveBayes())

            val ovrModel = oneVsRest.fit(train)
            val predictions = ovrModel.transform(test)

            // obtain evaluator.
            val accuracyEvaluator = new MulticlassClassificationEvaluator()
              .setMetricName("accuracy")
            val precisionEvaluator = new MulticlassClassificationEvaluator()
              .setMetricName("weightedPrecision")

            // compute the classification error on test data.
            val accuracy = accuracyEvaluator.evaluate(predictions)
            val precision = precisionEvaluator.evaluate(predictions)
            println(s"Accuracy  = $accuracy")
            println(s"Precision = $precision")

            "predictions:"
            predictions.show(1000, false)

            sc.stop()
          case Failure(e) =>
            throw e
        }
    }
}