package wedt

import java.io.File

import org.apache.spark.rdd.RDD
import wedt.WEDT.{logger, sparkContext}

import scala.util.{Failure, Random, Success, Try}

class DataProvider(path: String) extends Configuration {

  def prepareRddPerClass(perClass: Int): RDD[TaggedText] = {

    val subdirs = new File(path + "/").listFiles()

    val df = subdirs.map(subdir => {

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
          logSpark("number of loaded files: " + plainText.count())

          val baseDf = plainText
            .map(e => {
              e._3
                //.split("(Newsgroup|From):.*((.)*\\n){0,11}(From|Subject|Organization|: ===.*).*")
                .split("From:")
                .filter(e => e != "")
                .map(f => TaggedText(e._1, e._2, f))
            })

          val numberOfSubclasses = baseDf.count
          val numberOfRowsInASubclass = perClass/numberOfSubclasses

          val sample =
            baseDf.map(e =>
              Random.shuffle(e.toList).take(numberOfRowsInASubclass.toInt))
              .flatMap(e => e)
          logSpark(s"${subdir.getPath} gave ${sample.count} rows")
          sample.persist
        case Failure(e) =>
          //todo: zrobic porzadne logowanie
          logSpark(s"Could not load files from the path: $path")
          sparkContext.stop()
          throw e
      }
    }).reduce((a,b) => a.union(b))
      .persist
    logSpark(s"prepareRddPerClass has ${df.count} rows")
    df
  }
  def prepareRddPerSubclass(perSubclass: Int): RDD[TaggedText] = {

    val subdirs = new File(path + "/").listFiles()

    val df = subdirs.map(subdir => {

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
          logSpark("dataprovider: number of loaded files: " + plainText.count())

          val baseDf = plainText
            .map(e => {
              e._3
                //.split("((Newsgroup|From):.*((.)*\\n){0,11}(From|Subject|Organization|: ===.*).*)")
                .split("From:")
                .filter(e => e != "")
                .map(f => TaggedText(e._1, e._2, f))
            })

          val sample =
            baseDf.map(e =>
              Random.shuffle(e.toList).take(perSubclass))
              .flatMap(e => e)
          logSpark(s"${subdir.getPath} gave ${sample.count} rows")
          sample.persist
        case Failure(e) =>
          //todo: zrobic porzadne logowanie
          logSpark(s"Could not load files from the path: $path")
          sparkContext.stop()
          throw e
      }
    }).reduce((a,b) => a.union(b))
      .persist
    logSpark(s"prepareRddPerSubclass has ${df.count} rows")
    df
  }
}
