package wedt

import java.io._
import java.nio.file.{Files, Paths}
import java.util.Calendar

import org.apache.spark.ml.PipelineModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object ReadWriteToFileUtils {

  private val now = Calendar.getInstance()

  def saveModel(obj: Any, path: String): String = {
    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)

    oos.writeObject(obj)
    oos.flush()
    oos.close()

    val bytes = baos.toByteArray
    val _path = path + now
      .getTime
      .toString
      .replace(" ", "_")
      .replace(":", "_")
    Files.createFile(Paths.get(_path))
    val out = new FileOutputStream(_path)
    out.write(bytes)
    _path
  }
  def loadModel[T](path: String): T = {
    val byteArray = Files.readAllBytes(Paths.get(path))
    val bis = new ByteArrayInputStream(byteArray)
    val ois = new ObjectInputStream(bis)

      ois.readObject.asInstanceOf[T]
  }
}