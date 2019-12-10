import org.apache.spark.{SparkConf, SparkContext}

object TextClassifier {

  def main(args: Array[String]) {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val textFiles = sc.wholeTextFiles("src/main/resources/sport/football/*")
    println("liczba plikow: " + textFiles.count())

    textFiles.foreach(e => println(e._1))
    sc.stop()
  }

}