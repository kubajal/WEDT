package wedt

import java.nio.file.Files

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SerializationTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  import sqlContext.implicits._

  "Serialization" should "work on PipelineModel" in {

    val dataProvider = new DataProvider("resources/tests/*")
    val df = dataProvider.prepareRdd1(100)
      .toDF("firstLevelLabel", "secondLevelLabel", "features_0")

    val Array(trainDf, validateDf) = df.randomSplit(Array(0.7, 0.3))

    val mlc = new MultilayerClassifier(
      new NaiveBayes(),
      (for {i <- 1 to 20} yield new NaiveBayes()).toList,
      "bayes"
    )
    val trainedModel = new TextPipeline(mlc, 300).fit(trainDf)
    val validationResult = trainedModel.transform(validateDf)
    validationResult.map(e => (
      e.getAs[String]("features_0")
        .take(100)
        .replace("\n", "")
        .replace("\r", ""),
      e.getAs[Double]("prediction"),
      e.getAs[Double]("label")))
      .show(numRows = 100, truncate = false)

    val path = ReadWriteToFileUtils.saveModel(trainedModel, "tmp/" + trainedModel.stages.last.uid)
    val loadedModel = ReadWriteToFileUtils.loadModel[PipelineModel](path)
    val result = loadedModel.transform(validateDf)
  }


  "Serialization" should "work on other serializable classes" in {

    class Test(val num: Int, val str: String, val d: Double) extends Serializable {
      def numTest(): Int = num + 1
      def strTest(): String = str + "test"
      def dTest(): Double = d + 0.5
    }

    val test = new Test(1, "a", 1.0)

    val path = ReadWriteToFileUtils.saveModel(test, "tmp/" + "test.obj")
    val loadedTest = ReadWriteToFileUtils.loadModel[Test](path)
    assert(loadedTest.numTest() == 2)
    assert(loadedTest.strTest() == "atest")
    assert(loadedTest.dTest() == 1.5)
  }
}
