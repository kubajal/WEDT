package wedt

import java.io._
import java.nio.file.{Files, Paths}
import java.util.Calendar

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import wedt.WEDT.sparkContext

object ReadWriteToFileUtils {

  private val now = Calendar.getInstance()

  def saveModel(model: PipelineModel): String = {
    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)

    oos.writeObject(model)
    oos.flush()
    oos.close()

    val bytes = baos.toByteArray
    val _path = "models/" + model.stages.last.uid + now
      .getTime
      .toString
      .replace(" ", "_")
      .replace(":", "_")
    Files.createFile(Paths.get(_path))
    val out = new FileOutputStream(_path)
    out.write(bytes)
    _path
  }
  def loadModel(path: String): PipelineModel = {
    val byteArray = Files.readAllBytes(Paths.get(path))
    val bis = new ByteArrayInputStream(byteArray)
    val ois = new ObjectInputStream(bis)

      ois.readObject.asInstanceOf[PipelineModel]
  }
}