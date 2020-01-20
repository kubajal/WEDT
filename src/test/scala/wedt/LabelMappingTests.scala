package wedt

import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LabelMappingTests extends AnyFlatSpec with Matchers with Configuration {

  sparkContext.setLogLevel("ERROR")

   "Text classifier" should "split text data from a file into single e-mails" in {

    val textClassifier = new OneVsRest().setClassifier(new LogisticRegression())
    val rdd = WEDT.prepareRdd("src/main/resources/tests/*")
    val result = rdd.collect
    val counts = result
      .groupBy(e => e.firstLevelLabelValue)
    assert(counts.get(0.0).get.length == 10)
    assert(counts.get(1.0).get.length == 10)
   assert(counts.get(2.0).get.length == 10)
   assert(counts.get(3.0).get.length == 10)

  }

  "Each row" should "be correctly labeled according to path" in {

    val rdd = WEDT.prepareRdd("src/main/resources/tests/*")
    rdd.collect
      .foreach(e => {
        e.firstLevelLabelValue should be (WEDT.firstLevelLabelsMapping(e.firstLevelLabel))
        e.secondLevelLabelValue should be (WEDT.secondLevelLabelsMapping(e.secondLevelLabel))
      })
    WEDT.firstLevelLabelsMapping.size should be (4)
    WEDT.secondLevelLabelsMapping.size should be (4)
    WEDT.firstLevelLabelsMapping should contain ("soc" -> 0.0)
    WEDT.firstLevelLabelsMapping should contain ("alt" -> 2.0)
    WEDT.firstLevelLabelsMapping should contain ("comp" -> 1.0)
    WEDT.firstLevelLabelsMapping should contain ("rec" -> 3.0)
    WEDT.secondLevelLabelsMapping should contain ("atheism.txt" -> 0.0)
    WEDT.secondLevelLabelsMapping should contain ("graphics.txt" -> 1.0)
    WEDT.secondLevelLabelsMapping should contain ("sport.baseball.txt" -> 2.0)
    WEDT.secondLevelLabelsMapping should contain ("religion.christian.txt" -> 3.0)
  }
}
