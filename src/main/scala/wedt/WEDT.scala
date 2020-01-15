package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

object WEDT extends App with Configuration {
  var firstLevelLabelsMapping: Map[String, Double] = _
  var secondLevelLabelsMapping: Map[String, Double] = _

  //                                labelidx  text    label
  def prepareRdd(path: String): RDD[TaggedText] = {

    val plainTextTry = Try(sparkContext.wholeTextFiles(path))
    plainTextTry match {
      case Success(textData) =>

        // wyciaganie sciezki
        val plainText1 = textData
          .map(e => (e._1.split("/").takeRight(1).reduce((a,b) => a+"/"+b), e._2))
        val plainText2 = plainText1
          .map(e => (e._1.split("\\."), e._2))
        val plainText = plainText2
          .map(e => (e._1.head, e._1.takeRight(e._1.length-1).reduce((a,b) => a+"."+b), e._2))

        //todo: zrobic porzadne logowanie
        println("liczba wczytanych plikow: " + plainText.count())

        firstLevelLabelsMapping =  plainText
          .map(e => e._1)
          .distinct()
          .zipWithIndex
          .map(e => (e._1, e._2.toDouble))
          .collect()
          .toList
          .toMap

        secondLevelLabelsMapping =  plainText
          .map(e => e._2)
          .zipWithIndex
          .map(e => (e._1, e._2.toDouble))
          .collect()
          .toList
          .toMap

        plainText
          .map(e => (e._1, firstLevelLabelsMapping(e._1), e._2, secondLevelLabelsMapping(e._2), e._3))
          .flatMap(e => e._5
            .split("From:")
            .filter(e => e != "")
            .map(f => TaggedText(e._1, e._2, e._3, e._4, f)))
      case Failure(e) =>
        //todo: zrobic porzadne logowanie
        println(s"Could not load files from the path: $path")
        sparkContext.stop()
        throw e
    }
  }

  override def main(args: Array[String]): Unit = {

    import sqlContext.implicits._

    val textClassifier = new OneVsRestClassifier(new LogisticRegression())
    val path = if(args.length == 0) "resources/20-newsgroups/*" else args.head
//    val path = if(args.length == 0) "resources/sport/football/*" else args.head

    val df = this.prepareRdd(path)
      .toDF()
      .select("firstLevelLabelValue", "text")
      .withColumnRenamed("firstLevelLabelValue", "label")
      .withColumnRenamed("text", "features_0")

    val Array(train, validate) = pipeline.fit(df)
      .transform(df)
      .select("label", "features", "features_0")
      .randomSplit(Array(0.8, 0.2))

    println("schemat: ")
    train.printSchema()

    train.select("label")
        .groupBy("label")
    textClassifier.learn(train)
    textClassifier.test(validate)
  }
}
