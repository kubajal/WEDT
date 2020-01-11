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
    val dfTry = textClassifier.prepareRdd("resources/tests/*")
    dfTry match {
      case Success(df) =>
        val result = df.collect
        val counts = result
            .groupBy(e => e._1)

        assert(counts.get(0.0).get.length == 20)
        assert(counts.get(1.0).get.length == 20)
        result
      case Failure(e) =>
        throw e
    }
  }
}
