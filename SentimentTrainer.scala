import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


object SentimentTrainer {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Spark Sentiment")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    val twitterPath = args(0)

    val twitterTrainPath = args(0) + "/train.csv"
    val twitterTestPath = args(0) + "/test.csv"

    val twitterData = readTwitterData(twitterPath, spark)

    val tokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized All")
      .setPattern("\\s+")

    val hashing= new HashingTF()

    val classifier = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)


    /*
    assemble pipeline
     */

    val pipe = new Pipeline()
      .setStages(Array(tokenizer,
        hashing,
        classifier
      ))

    //val model = pipe.fit(twitterData)

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashing.numFeatures, Array(10, 100, 1000))
      .addGrid(classifier.regParam, Array(0.1, 0.01))
      .addGrid(classifier.tol, Array(1e-20, 1e-10, 1e-5))
      .addGrid(classifier.maxIter, Array(100, 200, 300))
      .build()
    //val model=pipe.fit(twitterData,paramGrid)

    val cv = new CrossValidator()
      .setEstimator(pipe)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = cv.fit(twitterData)
    //    val tr = model.transform(twitterData).select("Preprocessed", "Sentiment", "probability", "prediction")
//    tr.take(10).foreach(println)

  }

  def readTwitterData(path: String, spark: SparkSession) = {

    val data = spark.read.format("csv")
      .option("header", "true")
      .load(path)

    val preprocess: String => String = {
      _.replaceAll("((.))\\1+","$1")
    }
    val preprocessUDF = udf(preprocess)

    val newCol = preprocessUDF.apply(data("SentimentText"))
    data.withColumn("Preprocessed", newCol)
      .select("ItemID","Sentiment","Preprocessed")

  }
}


