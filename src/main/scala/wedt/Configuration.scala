package wedt

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.{SQLContext, SparkSession}
import wedt.WEDT.sparkContext

trait Configuration {

  val conf = new SparkConf()
  conf.setMaster("local")
  conf.setAppName("WEDT")
  val sparkSession: SparkSession = SparkSession
    .builder
    .config(conf)
    .getOrCreate()
  val sparkContext: SparkContext = sparkSession.sparkContext
  sparkContext.setLogLevel("ERROR")
  val defaultPath = "resources/20-newsgroups/*"
  val sqlContext: SQLContext = sparkSession.sqlContext
  val pipeline: TextPipeline = new TextPipeline()
  var model: Model[_] = _

  val layers: Array[Int] = Array[Int](4, 5, 4, 3)
}