package wedt

import org.apache.spark.rdd.RDD

object DataProvider extends Configuration {

  import sqlContext.implicits._

  val rdd: RDD[TaggedText] = WEDT.prepareRdd("src/main/resources/manual-tests/*")

  val Array(train, validate, rest) = rdd
    .toDF()
    .withColumnRenamed("text", "features_0")
    .randomSplit(Array(0.8, 0.2, 0.0))

}
