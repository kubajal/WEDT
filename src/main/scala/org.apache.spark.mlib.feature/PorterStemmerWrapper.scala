package org.apache.spark.mlib.feature

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
import org.tartarus.snowball.ext.PorterStemmer

class PorterStemmerWrapper(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], PorterStemmerWrapper] with DefaultParamsWritable {

  override protected def createTransformFunc: Seq[String] => Seq[String] = { seqOfStrings => {
      val stemClass = Class.forName("org.tartarus.snowball.ext.PorterStemmer")
      val stemmer = stemClass.newInstance.asInstanceOf[PorterStemmer]

      seqOfStrings.map(e => {
        stemmer.setCurrent(e)
        stemmer.stem()
        stemmer.getCurrent
      })
    }
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType.sameType(ArrayType(StringType)),
            s"Input type must be ArrayType(StringType) but got $inputType.")
  }

  override protected def outputDataType: DataType = ArrayType(StringType)
}