name := "BAC"

version := "0.2"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"


//lazy val commit = "242d49584c6aa21d928db2552033661950f760a5"

//lazy val g = RootProject(uri(s"git:file:///home/luca/code/spark/#$commit"))

//lazy val root = project in file(".") dependsOn g


    