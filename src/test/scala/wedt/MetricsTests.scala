package wedt

import org.apache.spark.ml.classification.{NaiveBayes, OneVsRest}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MetricsTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  import sqlContext.implicits._

  "Metrics" should "work" in {

    val df = sparkSession.createDataFrame(
      (1.0, 1.0) ::
      (1.0, 1.0) ::
      (1.0, 3.0) ::
      (2.0, 3.0) ::
      (2.0, 2.0) ::
      (2.0, 2.0) ::
      (3.0, 3.0) ::
      (3.0, 3.0) ::
      (3.0, 1.0) :: Nil
    )
    val roc = MetricsCalculator.roc(df, 1.0 :: 2.0 :: 3.0 :: Nil)

    roc.foreach(e => println(e))
    println("---")
    roc.foreach(e => println(e))


  }
}
