
name := "WEDT"

version := "0.1"

scalaVersion := "2.12.5"


libraryDependencies += "com.novocode" % "junit-interface" % "0.11" % "test"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.1.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.0" % "test"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0-preview"
// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.0-preview"

// https://mvnrepository.com/artifact/org.apache.lucene/lucene-analyzers-common
libraryDependencies += "org.apache.lucene" % "lucene-analyzers-common" % "8.4.0"
// https://mvnrepository.com/artifact/org.apache.lucene/lucene-core
libraryDependencies += "org.apache.lucene" % "lucene-core" % "8.4.0"
