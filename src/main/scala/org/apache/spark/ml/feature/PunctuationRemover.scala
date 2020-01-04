package org.apache.spark.ml.feature

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
import org.tartarus.snowball.ext.PorterStemmer

class PunctuationRemover(override val uid: String) extends UnaryTransformer[Seq[String], Seq[String], PunctuationRemover] with DefaultParamsWritable {

  override protected def createTransformFunc: Seq[String] => Seq[String] = { seqOfStrings => {
      seqOfStrings
        .map(e => {
          e.replaceAll("^([.!?,:;\"\'\\(\\[\\>\\<\\-]*)|([.!?,:;\"\'\\)\\]\\<\\>\\-]*)$", "")})
        .filter(e => e.matches("^[a-zA-Z0-9':-]+$"))
    }
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType.sameType(ArrayType(StringType)),
            s"Input type must be ArrayType(StringType) but got $inputType.")
  }

  override protected def outputDataType: DataType = ArrayType(StringType)
}