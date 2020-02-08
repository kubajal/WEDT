
name := "WEDT"

version := "0.2"

scalaVersion := "2.11.12"

lazy val global = (project in file("."))
  .aggregate(
    classifier,
    scrapper
  )

resolvers += Resolver.jcenterRepo

lazy val classifier = project
  .settings(libraryDependencies ++= Seq(
    "com.novocode" % "junit-interface" % "0.11" % "test",
    "org.scalactic" %% "scalactic" % "3.1.0",
    "org.scalatest" %% "scalatest" % "3.1.0" % "test",
    "org.apache.spark" %% "spark-mllib" % "2.4.4",
    "org.apache.spark" %% "spark-core" % "2.4.4",
    "org.apache.lucene" % "lucene-analyzers-common" % "8.4.0",
    "org.apache.lucene" % "lucene-core" % "8.4.0",
    "edu.stanford.nlp" % "stanford-corenlp" % "3.9.2",
    "com.typesafe.akka" %% "akka-http"   % "10.1.11",
    "com.typesafe.akka" %% "akka-stream" % "2.5.26" // or whatever the latest version is
  ))
lazy val scrapper = project
  .settings(libraryDependencies ++= Seq(
    "org.mongodb.scala" %% "mongo-scala-driver" % "2.8.0",
    "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.10.2",
    "org.jsoup" % "jsoup" % "1.9.1")
  )

