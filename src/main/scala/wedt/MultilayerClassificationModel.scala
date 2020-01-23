package wedt

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Model, PredictionModel, Transformer}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

class MultilayerClassificationModel(_uid: String,
                                    val firstLevelClassifier: Model[_],
                                    val firstLevelIndexer: StringIndexerModel,
                                    val globalIndexer: StringIndexerModel,
                                    val secondLevelClassifiers: Map[String, (StringIndexerModel, NaiveBayesModel)])
  extends Model[MultilayerClassificationModel] {

  import Implicits._

  val secondLevelReverseIndexers: Map[String, IndexToString] = secondLevelClassifiers
    .map(e => {
      (e._1, new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predicted2ndLevelClass")
        .setLabels(e._2._1.labels))
    })
  val firstLevelReverseIndexer: IndexToString = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predicted1stLevelClass")
    .setLabels(firstLevelIndexer.labels)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val df = firstLevelClassifier.transform(dataset)

    val firstLevelDf = firstLevelReverseIndexer.transform(df)
      .persist

    logger.info(s"transform: got dataset of ${dataset.count} rows")

    firstLevelDf.customShow()

    val secondLevelDf = firstLevelIndexer.labels
      .map(e => (e, firstLevelDf
        .withColumnRenamed("prediction", "1stLevelPrediction")
        .withColumnRenamed("rawPrediction", "1stLevelrawPrediction")
        .withColumnRenamed("probability", "1stLevelprobability")
        .filter(f => f.getAs[String]("predicted1stLevelClass") == e).persist))
      .map(e => {
        logger.info("classifying 2nd level: " + e)
        e._2.select("predicted1stLevelClass", "firstLevelLabel", "secondLevelLabel")
            .show(false)
        val classifier = secondLevelClassifiers(e._1)._2
        val result = classifier.transform(e._2)
        secondLevelReverseIndexers(e._1).transform(result)
          .withColumnRenamed("prediction", "2ndLevelPrediction")
          .withColumnRenamed("rawPrediction", "2ndLevelrawPrediction")
          .withColumnRenamed("probability", "2ndLevelprobability")})
      .reduce((a, b) => a.union(b))
    secondLevelDf
      .persist
  }

  override def copy(extra: ParamMap): MultilayerClassificationModel = {
    new MultilayerClassificationModel(_uid,
                                      firstLevelClassifier,
                                      firstLevelIndexer,
                                      globalIndexer,
                                      secondLevelClassifiers)
  }

  override def transformSchema(schema: StructType): StructType = {
    val structfield = StructField(name = "prediction", DoubleType, nullable = false, metadata = null)
    StructType(Array(structfield) ++ schema.fields)
  }

  override val uid: String = _uid
}
