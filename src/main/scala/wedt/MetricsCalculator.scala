package wedt

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

object MetricsCalculator extends Configuration {

  def roc(df: Dataset[Row], labels: List[Double]): List[(Double, Double)] = {
    import sqlContext.implicits._

    labels.map(l => {
      val rdd = df.rdd
        .map(row =>  {
          val prediction = row.getAs[Double]("_1")
          val label = row.getAs[Double]("_2")
            if(prediction == l && label == l)
              (1.0, 1.0)
            else if(prediction != l && label == l)
              (0.0, 1.0)
            else if(prediction != l && label != l)
              (0.0, 0.0)
            else if(prediction == l && label != l)
              (1.0, 0.0)
            else (0.0, 0.0)
        })
      val metrics = new BinaryClassificationMetrics(rdd)
      (l, metrics.areaUnderROC())
    })
  }

}
