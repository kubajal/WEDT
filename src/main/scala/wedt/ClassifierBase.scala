package wedt

import org.apache.spark.SparkContext
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

import scala.util.{Failure, Success, Try}

trait ClassifierBase extends Configuration {

  import sqlContext.implicits._

  def learn(train: DataFrame): Unit

  def test(df: DataFrame): Unit = {

    val predictions = model.transform(df)

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val accuracy = accuracyEvaluator.evaluate(predictions)
    val precision = precisionEvaluator.evaluate(predictions)
    println(s"Accuracy  = $accuracy")
    println(s"Precision = $precision")

    predictions
      .select("label", "prediction", "features_0")
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"), row.getAs[String]("features_0").take(100) + "..."))
      .show(false)
  }
}