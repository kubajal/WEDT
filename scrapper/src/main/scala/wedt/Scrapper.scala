package wedt

import scala.concurrent.duration._
import org.mongodb.scala.{Completed, Document, MongoClient, MongoCollection, MongoDatabase, Observer, Subscription}

object Scrapper extends App{
  override def main(args: Array[String]): Unit = {
    val mongoClient: MongoClient = MongoClient("mongodb://localhost:27017/")
    val database: MongoDatabase = mongoClient.getDatabase("test")
    val collection: MongoCollection[Document] = database.getCollection("test_collection")
    val count = collection.countDocuments
    count.subscribe((l: Long) => println(l))
    
    scala.concurrent.Await.result(count.toFuture(), 5 seconds)
    //val response = scala.io.Source.fromURI("https://content.guardianapis.com/search?q=football&api-key=6bd52fa5-a2db-4112-ada1-ba49f6e2d407").mkString
  }
}
