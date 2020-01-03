import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
import org.apache.spark.mlib.feature.PorterStemmerWrapper
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object TextClassifier {

  def main(args: Array[String]) {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val plainText = sc.wholeTextFiles("src/main/resources/sport/football/*")
    println("liczba plikow: " + plainText.count())

    val sqlContext = SparkSession.builder.getOrCreate().sqlContext
    import sqlContext.implicits._
    val df = plainText
      .map(e => e._2)
      .toDF()

    val tokenizer = new Tokenizer()
      .setInputCol("value")
      .setOutputCol("value1")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("value1")
      .setOutputCol("value2")
    val stemmer = new PorterStemmerWrapper("stemmer")
      .setInputCol("value2")
      .setOutputCol("value3")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, stemmer))

    val result = pipeline.fit(df)

    result
      .transform(df)
      .select("value3")
      .show(false)

    sc.stop()
  }

}