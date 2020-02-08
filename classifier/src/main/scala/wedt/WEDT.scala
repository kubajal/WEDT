package wedt

import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

object Scrapper extends App {

  override def main(args: Array[String]): Unit = {

    val response = scala.io.Source.fromURL("https://content.guardianapis.com/search?q=football&api-key=6bd52fa5-a2db-4112-ada1-ba49f6e2d407").mkString
    println(response)

  }

}