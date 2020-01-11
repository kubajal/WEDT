package wedt

import org.apache.spark.{SparkConf, SparkContext}

object WEDT extends App {

  private val defaultPath = "resources/20-newsgroups/*"

  override def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc: SparkContext = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val textClassifier = new TextClassifier(sc)
    val path = if(args.length == 0) "resources/20-newsgroups/*" else args.head
    textClassifier.run(path)
  }
}
