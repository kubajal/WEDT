package wedt

import java.io.File

import org.apache.spark.rdd.RDD
import wedt.WEDT.{logger, sparkContext}

import scala.util.{Failure, Random, Success, Try}

class DataProvider(path: String, train: Double, validate: Double) extends Configuration {

  def prepareRdd(numberOfRowsInAClass: Int): RDD[TaggedText] = {

    val subdirs = new File(path + "/").listFiles()

    subdirs.map(subdir => {

      val plainTextTry = Try(sparkContext.wholeTextFiles(subdir.getPath))
      plainTextTry match {
        case Success(textData) =>

          // wyciaganie sciezki
          val plainText1 = textData
            .map(e => (e._1.split("/").takeRight(1).reduce((a,b) => a+"/"+b), e._2))
          val plainText2 = plainText1
            .map(e => (e._1.split("\\."), e._2))
          val plainText = plainText2
            .map(e => (e._1.head, e._1.takeRight(e._1.length-1).reduce((a,b) => a+"."+b), e._2))

          //todo: zrobic porzadne logowanie
          logger.info("liczba wczytanych plikow: " + plainText.count())

          val baseDf = plainText
            .map(e => {
              e._3
                .split("((Newsgroup|From):.*((.)*\\n){0,11}(From|Subject|Organization|: ===.*).*)")
              //.split("From:")
                .filter(e => e != "")
                .map(f => TaggedText(e._1, e._2, f))
            })

          val numberOfSubclasses = baseDf.count
          val numberOfRowsInASubclass = numberOfRowsInAClass/numberOfSubclasses

          val sample =
            baseDf.map(e =>
              Random.shuffle(e.toList).take(numberOfRowsInASubclass.toInt))
            .flatMap(e => e)
          println(s"${subdir.getPath} gave ${sample.count} rows")
          sample
        case Failure(e) =>
          //todo: zrobic porzadne logowanie
          logger.info(s"Could not load files from the path: $path")
          sparkContext.stop()
          throw e
      }
    }).reduce((a,b) => a.union(b))
      .persist
  }

  import sqlContext.implicits._

  val rdd: RDD[TaggedText] = prepareRdd(500)

  private val counts = rdd.map(e => e.secondLevelLabel)
    .countByValue.toList
  println(s"counts of each secondLevelLabel: $counts")

  val Array(trainDf, validateDf, restDf) = rdd
    .toDF()
    .withColumnRenamed("text", "features_0")
    .randomSplit(Array(train, validate, Math.max(1 - train - validate, 0)))
}
