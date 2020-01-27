package wedt

import java.io.{ByteArrayOutputStream, File, ObjectOutputStream, PrintWriter}

import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import wedt.WEDT.{sparkContext, sqlContext}

class MultilayerClassifier[+M <: NaiveBayes](firstLevelOvrClassifier: M,
                                             secondLevelOvrClassifiers: List[M],
                                             _uid: String,
                                             subclassesVocabSize: Int)
  extends Estimator[MultilayerClassificationModel] {

  import Implicits._

  override val uid: String = _uid

  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains("features"), s"Column features does not exist.")
    schema
  }

  override def fit(dataset: Dataset[_]): MultilayerClassificationModel = {

    val df = dataset.asInstanceOf[Dataset[Row]]

    val firstLevelIndexer: StringIndexerModel = new StringIndexer()
      .setInputCol("firstLevelLabel")
      .setOutputCol("label")
      .fit(df)

    val firstLevelDataset = firstLevelIndexer.transform(df)

    val pm = firstLevelOvrClassifier.extractParamMap()

    logSpark(s"multi: fitting 1 level using ${df.count} rows")

    val cv = new CrossValidator()
      .setEstimator(firstLevelOvrClassifier)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setNumFolds(5)
      .setEstimatorParamMaps(Array(pm))

    val firstLevelClassifier = cv
        .fit(firstLevelDataset)

    val secondLevelClassifiers: Map[String, (StringIndexerModel, PipelineModel)] = firstLevelIndexer.labels
      .zip(secondLevelOvrClassifiers)
      .map(e => {
        val subset = df.filter(f => f.getAs[String]("firstLevelLabel") == e._1).persist
        val indexer = new StringIndexer()
          .setInputCol("secondLevelLabel")
          .setOutputCol("label")
          .fit(subset)
        val indexedSubset = indexer.transform(subset.drop("features", "features_1", "features_2", "features_3", "features_4", "features_5", "labels"))
        logSpark(s"multi: fitting ${e._1} pipeline using ${indexedSubset.count} rows")
        val cv = new CrossValidator()
          .setEstimator(e._2)
          .setEvaluator(new MulticlassClassificationEvaluator())
          .setNumFolds(5)
          .setEstimatorParamMaps(Array(pm))
        val pipeline = new TextPipeline(cv, subclassesVocabSize)
          .fit(indexedSubset)
        logSpark("multi: fitting 2nd level: " + e._1 + " using " + indexedSubset.count() + " rows")
        e._1 -> (indexer, pipeline)
      }).toMap

    val globalIndexer: StringIndexerModel = new StringIndexer()
      .setInputCol("secondLevelLabel")
      .setOutputCol("label")
      .fit(df)

    new MultilayerClassificationModel(_uid + "_model", firstLevelClassifier, firstLevelIndexer, globalIndexer, secondLevelClassifiers)
  }

  override def copy(extra: ParamMap): Estimator[MultilayerClassificationModel] = ???
}