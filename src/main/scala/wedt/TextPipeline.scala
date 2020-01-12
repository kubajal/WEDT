package wedt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._

class TextPipeline extends Pipeline {

  private val tokenizer = new Tokenizer()
    .setInputCol("features_0")
  private val stopWordsRemover = new StopWordsRemover()
    .setInputCol("features_1")
  private val punctuationRemover = new PunctuationRemover("punctuationRemover")
    .setInputCol("features_2")
  private val stemmer = new PorterStemmerWrapper("stemmer")
    .setInputCol("features_3")
  private val tf = new HashingTF()
    .setInputCol("features_4")
  private val idf = new IDF()
    .setInputCol("features_5")

  tokenizer.setOutputCol(stopWordsRemover.getInputCol)
  stopWordsRemover.setOutputCol(punctuationRemover.getInputCol)
  punctuationRemover.setOutputCol(stemmer.getInputCol)
  stemmer.setOutputCol(tf.getInputCol)
  tf.setOutputCol(idf.getInputCol)
  idf.setOutputCol("features")

  this.setStages(Array(tokenizer, stopWordsRemover, punctuationRemover, stemmer, tf, idf))
}
