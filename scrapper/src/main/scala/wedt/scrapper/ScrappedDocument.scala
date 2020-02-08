package wedt.scrapper

import org.mongodb.scala.Document
import org.mongodb.scala.bson.BsonDocument

class ScrappedDocument(url: String, value: String) extends Document(BsonDocument(s"""{"url": "$url", "value": "$value"}"""))