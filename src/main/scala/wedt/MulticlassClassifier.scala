package wedt

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

import scala.util.{Failure, Success, Try}

class MulticlassClassifier extends ClassifierBase {

  //todo: dorobic perceptron (czyli nie trzeba uzywac OneVsRest)
  override def learn(train: DataFrame): Unit = {
    val classifier = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    model = classifier.fit(train)
  }
}