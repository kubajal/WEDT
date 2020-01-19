package wedt

import org.apache.spark.sql.DataFrame

object Implicits extends Configuration {

  implicit def customShow(df: DataFrame): dfWithShow = dfWithShow(df)

  case class dfWithShow(df: DataFrame) {
    def customShow(): Unit = {
      import sqlContext.implicits._
      df.map(e => (
        e.getAs[String]("features_0")
          .take(100)
          .replace("\n", "")
          .replace("\r", ""),
        e.getAs[Double]("prediction"),
        e.getAs[Double]("label")))
        .show(numRows = 100, truncate = false)
    }
  }
}
