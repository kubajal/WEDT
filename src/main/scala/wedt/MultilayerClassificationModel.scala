package wedt

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Model, PredictionModel, Transformer}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
  import wedt.WEDT.{firstLevelLabelsMapping, log, sqlContext}

class MultilayerClassificationModel(_uid: String,
                                    firstLevelClassifier: CrossValidatorModel,
                                    secondLevelClassifiers: Map[Double, CrossValidatorModel])
  extends Model[MultilayerClassificationModel] {

  import sqlContext.implicits._

  override def transform(dataset: Dataset[_]): DataFrame = {
    val firstLevelDf = firstLevelClassifier.transform(dataset)
      .drop("prediction")
      .drop("rawPrediction")
      .drop("probability")
      .drop("label")
      .withColumnRenamed("secondLevelLabelValue", "label")
      .persist

    log.info(s"transform: got dataset of ${dataset.count} rows")

    firstLevelDf.show(false)

    val secondLevelDf = firstLevelLabelsMapping.values
      .map(e => (e, firstLevelDf
        .where($"firstLevelLabelValue" <=> e)))
      .map(e => {
        log.info("classifying 2nd level: " + e)
        secondLevelClassifiers(e._1).transform(e._2)})
      .reduce((a, b) => a.union(b))

    secondLevelDf
  }

  override def copy(extra: ParamMap): MultilayerClassificationModel = {
    new MultilayerClassificationModel(_uid,
                                      firstLevelClassifier,
                                      secondLevelClassifiers)
  }

  override def transformSchema(schema: StructType): StructType = {
//    require(schema.fieldNames.contains("firstLevelLabelValue"),
//      s"Column firstLevelLabelValue does not exist. Available field names: ${schema.fieldNames.toList}.")
    val structfield = StructField(name = "prediction", DoubleType, nullable = false, metadata = null)
    StructType(Array(structfield) ++ schema.fields)
  }

  override val uid: String = _uid
}
