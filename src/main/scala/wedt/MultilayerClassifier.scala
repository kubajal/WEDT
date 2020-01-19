package wedt

import org.apache.spark.ml.{Estimator, PipelineStage, Predictor, Transformer}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import wedt.TextPipeline
import wedt.WEDT.{firstLevelLabelsMapping, sqlContext}

class MultilayerClassifier(firstLevelOvrClassifier: OneVsRest,
                           secondLevelOvrClassifiers: List[OneVsRest],
                           _uid: String)
  extends Estimator[MultilayerClassificationModel] {

  import sqlContext.implicits._

  override val uid: String = _uid

  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains("firstLevelLabelValue"), s"Column firstLevelLabelValue does not exist.")
    require(schema.fieldNames.contains("secondLevelLabelValue"), s"Column secondLevelLabelValue does not exist.")
    require(schema.fieldNames.contains("features"), s"Column features does not exist.")
    schema
  }

  override def fit(df: Dataset[_]): MultilayerClassificationModel = {

    val firstLevelDataset = df.withColumnRenamed("firstLevelLabelValue", "label")
    val secondLevelDataset = df.withColumnRenamed("secondLevelLabelValue", "label")

    val pm = firstLevelOvrClassifier.extractParamMap()

    val firstLevelClassifier: CrossValidatorModel =
    new CrossValidator()
      .setEstimator(firstLevelOvrClassifier)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setNumFolds(5)
      .setEstimatorParamMaps(Array(pm))
      .fit(firstLevelDataset)

    val secondLevelClassifiers: Map[Double, CrossValidatorModel] = firstLevelLabelsMapping.values
      .zip(secondLevelOvrClassifiers)
      .map(e => (e._1 -> df.where($"firstLevelLabelValue" <=> e._1), e._2))
      .map(e => {
        val ovrClassifier = e._2
        val cv = new CrossValidator()
          .setEstimator(ovrClassifier)
          .setEvaluator(new MulticlassClassificationEvaluator())
          .setEstimatorParamMaps(Array(pm))
          .setNumFolds(5)
        e._1._1 -> cv.fit(secondLevelDataset)
      }).toMap

    new MultilayerClassificationModel(_uid + "_model", firstLevelClassifier, secondLevelClassifiers)
  }

  override def copy(extra: ParamMap): Estimator[MultilayerClassificationModel] = ???
}