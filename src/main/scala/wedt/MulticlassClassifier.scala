package wedt

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}

import scala.util.{Failure, Success, Try}

class MulticlassClassifier(val sc: SparkContext, val pipeline: Pipeline) {
  //todo: dorobic perceptron (czyli nie trzeba uzywac OneVsRest)

  private val defaultPath = "resources/20-newsgroups/*"
  var ovrModel: OneVsRestModel = _

  val sqlContext: SQLContext = SparkSession.builder.getOrCreate().sqlContext
}