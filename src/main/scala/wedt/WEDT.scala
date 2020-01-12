package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

object WEDT extends App with Configuration {

  def prepareRdd(path: String): RDD[(Double, String)] = {

    val plainTextTry = Try(sparkContext.wholeTextFiles(path))
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
            .map(f => (e._3, f.take(100).replaceAll("[\n\r]", ""))))
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        println(s"Could not load files from the path: $path")
        sparkContext.stop()
        throw e
    }
  }

  override def main(args: Array[String]): Unit = {

    import sqlContext.implicits._

    val textClassifier = new OneVsRestClassifier(new NaiveBayes())
    val path = if(args.length == 0) "resources/20-newsgroups/*" else args.head

    val df = this.prepareRdd(path)
      .toDF()
      .withColumnRenamed("_1", "label")
      .withColumnRenamed("_2", "features_0")

    val Array(train, validate) = pipeline.fit(df)
      .transform(df)
      .select("label", "features", "features_0")
      .randomSplit(Array(0.8, 0.2))

    textClassifier.learn(train)
    textClassifier.test(validate)
  }
}
