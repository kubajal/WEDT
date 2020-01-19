package wedt

import java.util.Calendar

import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}
import wedt.WEDT.firstLevelLabelsMapping

import scala.util.{Failure, Success, Try}

object WEDT extends Configuration {
  var firstLevelLabelsMapping: Map[String, Double] = _
  var secondLevelLabelsMapping: Map[String, Double] = _

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
        log.info("liczba wczytanych plikow: " + plainText.count())

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
        log.info(s"Could not load files from the path: $path")
        sparkContext.stop()
        throw e
    }
  }
}
