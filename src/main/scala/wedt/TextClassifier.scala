package wedt

import org.apache.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark

import scala.util.{Failure, Success, Try}

class TextClassifier(val sc: SparkContext, val pipeline: Pipeline) {

  private val defaultPath = "resources/20-newsgroups/*"
  var ovrModel: OneVsRestModel = _

  val sqlContext: SQLContext = SparkSession.builder.getOrCreate().sqlContext
  import sqlContext.implicits._

  def test(): Unit = {

    val df = prepareRdd("resources/manual-tests/*")
      .map(e => e._2)
      .toDF()
      .withColumnRenamed("value", "features_0")
    val testData = pipeline.fit(df)
      .transform(df)

    val result = ovrModel.transform(testData)
    result
      .select("features_0", "prediction")
      .map(row => (row.getAs[String]("features_0").take(100) + "...", row.getAs[Double]("prediction")))
      .show(false)
  }

  def prepareRdd(path: String): RDD[(Double, String)] = {

    val plainTextTry = Try(sc.wholeTextFiles(path))
    plainTextTry match {
      case Success(plainText) =>

        //todo: zrobic porzadne logowanie
        println("liczba wczytanych plikow: " + plainText.count())

        plainText
          .zipWithIndex
          .map(e => (e._1._1, e._1._2, e._2.toDouble))
          .flatMap(e => e._2
            .split("From:")
            .filter(e => e != "")
            .map(f => (e._3, f.take(100))))
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        println(s"Could not load files from the path: $path")
        sc.stop()
        throw e
    }
  }

  def learn(rdd: RDD[(Double, String)]): Unit = {

    val sqlContext = SparkSession.builder.getOrCreate().sqlContext
    import sqlContext.implicits._

    val df = rdd.toDF()
      .withColumnRenamed("_1", "label")
      .withColumnRenamed("_2", "features_0")

    val Array(train, test) = pipeline.fit(df)
      .transform(df)
      .select("label", "features")
      .randomSplit(Array(0.8, 0.2))

    println("random split -- train: " + train.count + ", test: " + test.count)

    val oneVsRest = new OneVsRest().setClassifier(new NaiveBayes())

    ovrModel = oneVsRest.fit(train)
    val predictions = ovrModel.transform(test)

    // obtain evaluator.
//            val accuracyEvaluator = new MulticlassClassificationEvaluator()
//              .setMetricName("accuracy")
//            val precisionEvaluator = new MulticlassClassificationEvaluator()
//              .setMetricName("weightedPrecision")

    // compute the classification error on test data.
//            val accuracy = accuracyEvaluator.evaluate(predictions)
//            val precision = precisionEvaluator.evaluate(predictions)
//            println(s"Accuracy  = $accuracy")
//            println(s"Precision = $precision")

//            "predictions:"
//            predictions.show(1000, false)
    }

  def predict(text: String): Unit = {

    val sparkSession = SparkSession.builder.getOrCreate()

    val df = ovrModel.transform(
      sparkSession.createDataset(Array(text))
      .withColumnRenamed("value", "features"))
    df.show(false)
  }
}