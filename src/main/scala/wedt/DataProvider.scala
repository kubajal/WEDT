package wedt

import org.apache.spark.rdd.RDD

class DataProvider(path: String, train: Double, validate: Double) extends Configuration {

  import sqlContext.implicits._

  val rdd: RDD[TaggedText] = WEDT.prepareRdd(path + "/*")

  val Array(trainDf, validateDf, restDf) = rdd
    .toDF()
    .withColumnRenamed("text", "features_0")
    .randomSplit(Array(train, validate, Math.max(1 - train - validate, 0)))
}
