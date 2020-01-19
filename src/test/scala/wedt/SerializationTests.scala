package wedt

import java.nio.file.Files

import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SerializationTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

  import sqlContext.implicits._

  "Serialization" should "work" in {

    val rdd = WEDT.prepareRdd("resources/tests/*")
    rdd.collect
      .foreach(e => {
        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
      })

    val Array(train, validate) = rdd
      .toDF()
      .withColumnRenamed("text", "features_0")
      .randomSplit(Array(0.7, 0.3))

    val mlc = new MultilayerClassifier(
      new OneVsRest().setClassifier(new NaiveBayes()),
      (for {i <- 1 to 20} yield new OneVsRest().setClassifier(new NaiveBayes())).toList,
      "bayes"
    )
    val trainedModel = new TextPipeline(mlc).fit(train)
    val validationResult = trainedModel.transform(validate)
    validationResult.map(e => (
      e.getAs[String]("features_0")
        .take(100)
        .replace("\n", "")
        .replace("\r", ""),
      e.getAs[Double]("prediction"),
      e.getAs[Double]("label")))
      .show(numRows = 100, truncate = false)

    val path = ReadWriteToFileUtils.saveModel(trainedModel)
    val loadedModel = ReadWriteToFileUtils.loadModel(path)
    val result = loadedModel.transform(validate)
    import Implicits._
    result.customShow()
  }
}
