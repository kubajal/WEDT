package wedt

import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.Dataset

class TextPipeline(mlc: Estimator[_], vocabSize: Int) extends Pipeline with Configuration {

  private val stopWords = sparkContext.textFile("src/main/scala/wedt/stopwords.txt").collect.distinct
  println(s"list of stopwords: ${stopWords.toList}")

  private val tokenizer = new Tokenizer()
    .setInputCol("features_0")
  private val punctuationRemover = new PunctuationRemover("punctuationRemover")
    .setInputCol("features_1")
  private val stopWordsRemover = new StopWordsRemover()
    .setInputCol("features_2")
    .setStopWords(stopWords)
  println(s"stopWordsRemover stoplist: ${stopWordsRemover.stopWords}")
  private val stemmer = new PorterStemmerWrapper("stemmer")
    .setInputCol("features_3")
  private val vectorizer = new CountVectorizer()
    .setInputCol("features_4")
    .setVocabSize(vocabSize)
  private val idf = new IDF()
    .setInputCol("features_5")

  tokenizer.setOutputCol(punctuationRemover.getInputCol)
  punctuationRemover.setOutputCol(stopWordsRemover.getInputCol)
  stopWordsRemover.setOutputCol(stemmer.getInputCol)
  stemmer.setOutputCol(vectorizer.getInputCol)
  vectorizer.setOutputCol(idf.getInputCol)
  idf.setOutputCol("features")

  this.setStages(Array(tokenizer, punctuationRemover, stopWordsRemover, stemmer, vectorizer, idf, mlc))
}
