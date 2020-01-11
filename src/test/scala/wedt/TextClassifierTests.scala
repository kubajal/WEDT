package wedt

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.{Failure, Success}

class TextClassifierTests extends AnyFlatSpec with Matchers {

  "Text classifier" should "split text data from a file into single e-mails" in {

    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("WEDT")
    val sc: SparkContext = new SparkContext(conf)
    sc.setLogLevel("DEBUG")

    val textClassifier = new TextClassifier(sc)
    val dfTry = textClassifier.prepareDf("resources/tests/*")
    dfTry match {
      case Success(df) =>
        df.collect should have size 20
      case Failure(e) =>
        throw e
    }
  }

}
