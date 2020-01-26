package wedt

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType
import wedt.Implicits.logger
import wedt.WEDT.sqlContext

class SingleLayerClassifier(firstLevelOvrClassifier: Estimator[_],
                            _uid: String)
  extends Estimator[CrossValidatorModel] {

  import sqlContext.implicits._

  var indexer: StringIndexerModel = _

  override val uid: String = _uid

  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains("features"), s"Column features does not exist.")
    schema
  }

  override def fit(dataset: Dataset[_]): CrossValidatorModel = {

    val df = dataset.asInstanceOf[DataFrame]

    log.info(s"fit: got df of ${df.count} rows")

    val featuresNumber = df.head.getAs[org.apache.spark.ml.linalg.Vector]("features").size
    logger.info(s"single-bayes: number of features: ${featuresNumber}")

    indexer = new StringIndexer()
      .setInputCol("secondLevelLabel")
      .setOutputCol("label")
      .fit(df)
    val secondLevelDataset = indexer.transform(df)
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