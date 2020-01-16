package wedt

import java.util.Calendar

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import wedt.WEDT.firstLevelLabelsMapping

import scala.util.{Failure, Success, Try}

object WEDT extends App with Configuration {
  var firstLevelLabelsMapping: Map[String, Double] = _
  var secondLevelLabelsMapping: Map[String, Double] = _
  val time = Calendar.getInstance()

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

    println("poczatek: " + time.getTime.toString)

    import sqlContext.implicits._

    val path = if(args.length == 0) "resources/20-newsgroups/*" else args.head
//    val path = if(args.length == 0) "resources/sport/football/*" else args.head

    val df = this.prepareRdd(path)
      .toDF()
      .withColumnRenamed("text", "features_0")

    val pipeline = new TextPipeline()
      .fit(df)

    val firstLevelInput = df.withColumnRenamed("firstLevelLabelValue", "label")
    val secondLevelInput = df.withColumnRenamed("secondLevelLabelValue", "label")
    val lr = new LogisticRegression()
    val firstLevelOvrClassifier = new OneVsRest().setClassifier(lr)

    val firstLevelDataset = pipeline.transform(firstLevelInput)
    val secondLevelDataset = pipeline.transform(secondLevelInput)

    val paramMap = lr.extractParamMap()

    val firstLevelClassifier =
      new CrossValidator()
        .setEstimator(firstLevelOvrClassifier)
        .setEvaluator(new MulticlassClassificationEvaluator())
        .setNumFolds(5)
        .setEstimatorParamMaps(Array(paramMap))
        .fit(firstLevelDataset)

    val secondLevelClassifiers = firstLevelLabelsMapping.values
      .map(e => (e -> df.where($"firstLevelLabelValue" <=> e)))
      .map(e => {
        val ovrClassifier = new OneVsRest().setClassifier(new LogisticRegression())
        val cv = new CrossValidator()
          .setEstimator(ovrClassifier)
          .setEvaluator(new MulticlassClassificationEvaluator())
          .setEstimatorParamMaps(Array(paramMap))
          .setNumFolds(5)
        e._1 -> cv.fit(secondLevelDataset)
      }).toMap

    val a = df
    val aa = a
      .withColumnRenamed("firstLevelLabelValue", "label")
      .withColumnRenamed("text", "features_0")
    val bb = a
      .withColumnRenamed("secondLevelLabelValue", "label")
      .withColumnRenamed("text", "features_0")

    val aaa = pipeline
      .transform(aa)
    val firstLevel = firstLevelClassifier.transform(aaa)

    println("first level: ")
    firstLevel
      .drop("features_0")
      .show(false)

    val secondLevel = firstLevelLabelsMapping.values
      .map(e => (e, firstLevel
        .where($"firstLevelLabelValue" <=> e)))
      .map(e => secondLevelClassifiers(e._1).transform(e._2
        .drop("prediction")
        .drop("rawPrediction")
        .drop("label")
      .withColumnRenamed("secondLevelLabelValue", "label")))
      .reduce((a, b) => a.union(b))

    println("second level: ")
    secondLevel
      .drop("features_0")
      .show(false)

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")
    val accuracy = accuracyEvaluator.evaluate(secondLevel)
    val precision = precisionEvaluator.evaluate(secondLevel)
    println(s"Accuracy  = $accuracy")
    println(s"Precision = $precision")


    println("koniec: " + time.getTime.toString)
  }
}
