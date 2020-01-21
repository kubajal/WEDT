package wedt

import java.io.{ByteArrayOutputStream, ObjectOutputStream}

import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import wedt.WEDT.{firstLevelLabelsMapping, sparkContext, sqlContext}

class MultilayerClassifier[+M <: Estimator[_]](firstLevelOvrClassifier: M,
                           secondLevelOvrClassifiers: List[M],
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

    log.info(s"fit: got df of ${df.count} rows")

    val firstLevelDataset = df.withColumnRenamed("firstLevelLabelValue", "label")
    val secondLevelDataset = df.withColumnRenamed("secondLevelLabelValue", "label")

    val pm = firstLevelOvrClassifier.extractParamMap()

    log.info("fitting 1 level")
    val firstLevelClassifier: CrossValidatorModel =
      new CrossValidator()
        .setEstimator(firstLevelOvrClassifier)
        .setEvaluator(new MulticlassClassificationEvaluator())
        .setNumFolds(5)
        .setEstimatorParamMaps(Array(pm))
        .fit(firstLevelDataset)

    val secondLevelClassifiers: Map[Double, CrossValidatorModel] = firstLevelLabelsMapping.values
      .zip(secondLevelOvrClassifiers)
      .map(e => (e._1 -> secondLevelDataset.where($"firstLevelLabelValue" <=> e._1), e._2))
      .map(e => {
        val ovrClassifier = e._2
        log.info("fitting 2nd level: " + e._1._1)
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