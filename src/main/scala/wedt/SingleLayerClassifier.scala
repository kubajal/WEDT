package wedt

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import wedt.WEDT.{firstLevelLabelsMapping, sqlContext}

class SingleLayerClassifier(firstLevelOvrClassifier: OneVsRest,
                            _uid: String)
  extends Estimator[CrossValidatorModel] {

  import sqlContext.implicits._

  override val uid: String = _uid

  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains("secondLevelLabelValue"), s"Column secondLevelLabelValue does not exist.")
    require(schema.fieldNames.contains("features"), s"Column features does not exist.")
    schema
  }

  override def fit(df: Dataset[_]): CrossValidatorModel = {

    log.info(s"fit: got df of ${df.count} rows")

    val secondLevelDataset = df.withColumnRenamed("secondLevelLabelValue", "label")

    val pm = firstLevelOvrClassifier.extractParamMap()

    log.info("fitting 1 level")
    new CrossValidator()
      .setEstimator(firstLevelOvrClassifier)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setNumFolds(5)
      .setEstimatorParamMaps(Array(pm))
      .fit(secondLevelDataset)
  }

  override def copy(extra: ParamMap): Estimator[CrossValidatorModel] = ???
}