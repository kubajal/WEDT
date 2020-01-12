package wedt

import org.apache.log4j.LogManager
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark

import scala.util.{Failure, Success, Try}

class OneVsRestClassifier(override val classifier: Classifier[_,_,_]) extends ClassifierBase(classifier) {

  override def learn(train: DataFrame): Unit = {
    val oneVsRest = new OneVsRest().setClassifier(classifier)
    model = oneVsRest.fit(train)
  }
}