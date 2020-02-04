
name := "WEDT"

version := "0.2"

scalaVersion := "2.11.12"

lazy val global = (project in file("."))
  .aggregate(
    classifier,
    scrapper
  )

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
  ) )
lazy val scrapper = project
  .settings(libraryDependencies ++= Seq("com.typesafe.akka" %% "akka-actor" % "2.6.1",
    "com.typesafe.akka" %% "akka-cluster-tools" % "2.6.1",
    "com.typesafe.akka"     %% "akka-remote" % "2.6.1",
    "com.typesafe.akka" %% "akka-cluster-typed" % "2.6.1",
    "com.typesafe.play" %% "play-iteratees" % "2.6.1",
    "org.scala-js" %% "scalajs-library" % "1.0.0-RC2") )

