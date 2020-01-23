package wedt

import org.apache.spark.sql.DataFrame

object Implicits extends Configuration {

  implicit def customShow(df: DataFrame): dfWithShow = dfWithShow(df)

  case class dfWithShow(df: DataFrame) {
    def customShow(col: String = "prediction", col2: String = null): Unit = {
      import sqlContext.implicits._
      if(col2 == null){
        df.map(e => (
          e.getAs[String]("features_0")
            .take(20)
            .replace("\n", "")
            .replace("\r", ""),
          e.getAs[Double](col)))
          .show(numRows = 100, truncate = false)
      }
      else{
        df.map(e => (
          e.getAs[String]("features_0")
            .take(20)
            .replace("\n", "")
            .replace("\r", ""),
          e.getAs[Double](col),
          e.getAs[Double](col2)))
          .show(numRows = 100, truncate = false)
      }
    }
  }
}
