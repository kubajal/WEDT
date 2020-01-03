package org.apache.spark.mllib.feature

import org.apache.spark.mlib.feature.PorterStemmerWrapper
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PorterStemmerWrapperTests extends AnyFlatSpec with Matchers {

  private val stemmer = new PorterStemmerWrapper("uid")
    .setInputCol("value")
    .setOutputCol("result")

  private val sparkSession = SparkSession
    .builder()
    .appName("tests")
    .config("spark.master", "local")
    .getOrCreate()
  import sparkSession.implicits._

   "Porter stemmer" should "correctly stem words" in {
    val words = sparkSession.createDataset(Seq("Example"
        :: "weakness"
        :: "yields"
        :: "temptation"
        :: "are"
        :: "terrible"
        :: "temptations"
        :: "requires"
        :: "courage"
        :: "Wilde" :: Nil)
    )

     val result = stemmer
       .transform(words)
       .head
       .getAs[Seq[String]]("result")

     println(result)

     result should contain ("Exampl")
     result should contain ("weak")
     result should contain ("yield")
     result should contain ("temptat")
     result should contain ("ar")
     result should contain ("terribl")
     result should contain ("temptat")
     result should contain ("requir")
     result should contain ("courag")
     result should contain ("Wild")
  }
}
