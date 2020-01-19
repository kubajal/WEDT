package wedt.experiment

import org.apache.spark.rdd.RDD
import wedt.{Configuration, TaggedText, WEDT}

object DataProvider extends Configuration {

  import sqlContext.implicits._

  val rdd: RDD[TaggedText] = WEDT.prepareRdd("resources/20-newsgroups/*")

  val Array(train, validate, rest) = rdd
    .toDF()
    .withColumnRenamed("text", "features_0")
    .randomSplit(Array(0.2, 0.1, 0.7))

}
