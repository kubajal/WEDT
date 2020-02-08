package wedt.scrapper

import java.util.Date

import com.fasterxml.jackson.annotation.JsonProperty

case class Result(id: String,
                  @JsonProperty("type") resultType: String,
                  sectionId: String,
                  sectionName: String,
                  webPublicationDate: Date,
                  webTitle: String,
                  webUrl: String,
                  apiUrl: String,
                  isHosted: Boolean,
                  pillarId: String,
                  pillarName: String
                 )

case class Response(status: String,
                    userTier: String,
                    total: Long,
                    startIndex: Long,
                    pageSize: Long,
                    currentPage: Long,
                    pages: Long,
                    orderBy: String,
                    results: Array[Result])
case class Body(response: Response) {}
case  class Test(field: String) {}