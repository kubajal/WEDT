package org.apache.spark.ml.feature

import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PunctuationRemoverTests extends AnyFlatSpec with Matchers {

  private val punctuationRemover = new PunctuationRemover("uid")
    .setInputCol("value")
    .setOutputCol("result")

  private val sparkSession = SparkSession
    .builder()
    .appName("tests")
    .config("spark.master", "local")
    .getOrCreate()
  import sparkSession.implicits._

  "Punctuation remover" should "correctly remove punctuation at the beginning of a word" in {
    val testData = Seq(",a", ".b", "-c", ":d", ";e", "(f", "[g", "\"h", "'i")
    val words = sparkSession.createDataset(Seq(testData))

    val result = punctuationRemover
      .transform(words)
      .head
      .getAs[Seq[String]]("result")

    result should contain ("a")
    result should contain ("b")
    result should contain ("c")
    result should contain ("d")
    result should contain ("e")
    result should contain ("f")
    result should contain ("g")
    result should contain ("h")
    result should contain ("i")
  }

  "Punctuation remover" should "correctly remove punctuation at the end of a word" in {
    val testData = Seq("h,", "i.", "j-", "k:", "l;", "m)", "n]", "o\"", "p'")
    val words = sparkSession.createDataset(Seq(testData))

    val result = punctuationRemover
      .transform(words)
      .head
      .getAs[Seq[String]]("result")

    result should contain ("h")
    result should contain ("i")
    result should contain ("j")
    result should contain ("k")
    result should contain ("l")
    result should contain ("m")
    result should contain ("n")
    result should contain ("o")
    result should contain ("p")
  }

  "Punctuation remover" should "not remove punctuation in the middle of a word" in {
    val testData = Seq("a,a", "b.b", "c-c", "d:d", "e;e", "f(f", "g[g", "h)h", "i]i")
    val words = sparkSession.createDataset(Seq(testData))

    val result = punctuationRemover
      .transform(words)
      .head
      .getAs[Seq[String]]("result")

    result should contain ("a,a")
    result should contain ("b.b")
    result should contain ("c-c")
    result should contain ("d:d")
    result should contain ("e;e")
    result should contain ("f(f")
    result should contain ("g[g")
    result should contain ("h)h")
    result should contain ("i]i")
  }

  "Punctuation remover" should "remove multiple exclamation and question marks from the ending of a word" in {
    val testData = Seq("a?!?!?!!???", "b???", "c!!!", "d!!!\"")
    val words = sparkSession.createDataset(Seq(testData))

    val result = punctuationRemover
      .transform(words)
      .head
      .getAs[Seq[String]]("result")

    result should contain ("a")
    result should contain ("b")
    result should contain ("c")
    result should contain ("d")
  }
}
