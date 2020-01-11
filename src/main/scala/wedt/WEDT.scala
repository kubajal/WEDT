package wedt

import org.apache.spark.{SparkConf, SparkContext}

class WEDT extends App {
  override def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc: SparkContext = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val textClassifier = new TextClassifier(sc)


  }
}
