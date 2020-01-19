package wedt

import org.apache.log4j.Logger
import org.apache.spark.ml.feature._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import wedt.WEDT.{getClass, sparkContext}

trait Configuration {

  val conf = new SparkConf()
  conf.setMaster("local")
  conf.setAppName("WEDT")
  val sparkSession: SparkSession = SparkSession
    .builder
    .config(conf)
    .getOrCreate()
  val sparkContext: SparkContext = sparkSession.sparkContext
  val defaultPath = "resources/20-newsgroups/*"
  val sqlContext: SQLContext = sparkSession.sqlContext
  var model: Model[_] = _
  sparkContext.setLogLevel("INFO")
  val log: Logger = Logger.getLogger(getClass.getName)

  val layers: Array[Int] = Array[Int](4, 5, 4, 3)
}