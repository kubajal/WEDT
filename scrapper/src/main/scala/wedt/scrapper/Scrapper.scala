package wedt.scrapper

import java.io.StringWriter
import java.net.URL
import java.util

import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.jsoup.Jsoup
import org.jsoup.nodes.{Element, Node, TextNode}

import scala.reflect.ClassTag

object Scrapper extends App{


  def flatten(elements: util.List[Node]): Option[String] = {
    import scala.collection.JavaConverters._
    val x = elements.asScala
      .flatMap{
        case (n: Element) if n.tag.getName == "p" => n :: Nil
        case _ => Nil
    }.toList

    if(x.isEmpty)
      None
    else {
      val list3 = x
        .map(e => e.text)
        .filter(e => e != "" && e != null)
      val list4 = list3.reduce((a, b) => a + " " + b)
      Some(list4)
    }
  }

  override def main(args: Array[String]): Unit = {

    val apiKey = args(0)

//    val mongoClient: MongoClient = MongoClient("mongodb://localhost:27017/")
//    val database: MongoDatabase = mongoClient.getDatabase("test")
//    val collection: MongoCollection[Document] = database.getCollection("test_collection")
//    val count = collection.countDocuments
//    count.subscribe((l: Long) => println(l))
//    val insert = collection.insertMany(Seq(new ScrappedDocument("url1", "value1"), new ScrappedDocument("url2", "value2")))
//
//    insert.subscribe(new Observer[Completed] {
//      override def onNext(result: Completed): Unit = println(s"onNext: $result")
//      override def onError(e: Throwable): Unit = println(s"onError: $e")
//      override def onComplete(): Unit = println("onComplete")
//    })

    //scala.concurrent.Await.result(count.toFuture(), 5 seconds)
    //scala.concurrent.Await.result(insert.toFuture(), 5 seconds)
    //val count1 = collection.countDocuments
    //count1.subscribe((l: Long) => println(l))
    //scala.concurrent.Await.result(count1.toFuture(), 5 seconds)
    val mapper: ObjectMapper = new ObjectMapper()
    mapper.registerModule(DefaultScalaModule)
    val out = new StringWriter
    mapper.writeValue(out, Test("abc"))
    val json = out.toString
    println(json)
    val response = scala.io.Source.fromURL(s"https://content.guardianapis.com/search?q=football&api-key=$apiKey").mkString
    val result = mapper.readValue(response, classOf[Body])

    result.response.results.foreach(response => {
      val html = scala.io.Source.fromURL(response.webUrl).mkString

      val div = Jsoup.parse(html)
        .select(".content__article-body").first()

      for {
        text <- flatten(div.childNodes)
      } yield {
        println(text)
      }
    })

  }
}
