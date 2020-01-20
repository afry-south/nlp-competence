import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline, SparkNLP}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.{Source, StdIn}

object SparkApproachFinished {
  val spark: SparkSession = SparkNLP.start
  val pipeline: LightPipeline = PretrainedPipeline("explain_document_dl_fast", lang="en").lightModel
  // Below is a reason to hate Scala, hehe.
  import spark.implicits.{getClass => _, _}

  private def readMuellerReport(): Seq[(String, Int)] = {
    // Pro-tip: Use stream if you're gonna deploy code as library
    val muellerReportPath = getClass.getResourceAsStream("report_mueller.txt")
    val src = Source.fromInputStream(muellerReportPath)
    val muellerText = src.getLines.toList.filterNot(_.forall(_.isWhitespace)).zipWithIndex
    src.close

    muellerText
  }

  private def stepOne(): Unit = {
    val testData = spark.createDataFrame(Seq(
      (1, "Google has announced the release of a beta version of the popular TensorFlow machine learning library"),
      (2, "Donald John Trump (born June 14, 1946) is the 45th and current president of the United States")
    )).toDF("id", "text")

    val annotation = pipeline.transform(testData)

     annotation.show()
     StdIn.readLine("Press Enter to continue")
     annotation.explain()
     StdIn.readLine("Press Enter to continue")

     annotation.select($"sentence.result".as("sentences")).show(false)
     StdIn.readLine("Press Enter to continue")
     annotation.select($"token.result".as("tokens")).show(false)
     StdIn.readLine("Press Enter to continue")
     annotation.select($"ner.result".as("ner")).show(false)
     StdIn.readLine("Press Enter to continue")
     annotation.select($"entities.result".as("entities")).show(false)
     StdIn.readLine("Press Enter to continue")
  }

  def main(args: Array[String]): Unit = {
    lazy val muellerData: DataFrame = spark
      .createDataFrame(readMuellerReport())
      .toDF("text", "line") // TODO perhaps repartitioning would do good.

    StdIn.readLine("Press Enter to go to step 1")
    stepOne()

    StdIn.readLine("Press Enter to go to step 2")
    val annotatedMuellerData = pipeline.transform(muellerData)
    annotatedMuellerData.show()

    StdIn.readLine("Press Enter to go to step 3")
    stepThree(muellerData)

    StdIn.readLine("Press Enter to go to step 4")
    findCategories(annotatedMuellerData.select($"token.result".as("token")), spark)

    /**
     * Improvements we can implement
     * 1. Clean data using the pipelines available in SparkNLP (stopwords, stemming etc)
     * 2. Make the documents larger than a line, i.e. split by chapter etc
     * 3. Use BERT embeddings instead
     * 4. Add POS taggings etc
     */

    println("Finished! Let's go to non-spark! [Smile]")
  }

  def findCategories(muellerData: DataFrame, spark: SparkSession): Unit = {
    import spark.implicits.{getClass => _}
    val cv = new CountVectorizer().setOutputCol("rawFeatures").setInputCol("token") // A CountVectorizer, or HashVectorizer with correct output/inputcols
    val cvModel = cv.fit(muellerData) // fitTransform on these two lines
    val featureizedData = cvModel.transform(muellerData)
    val vocab = cvModel.vocabulary.toList
    val vocabBroadcast = spark.sparkContext.broadcast(vocab)

    val idf = new IDF().setOutputCol("features").setInputCol("rawFeatures")
    val idfModel = idf.fit(featureizedData)
    val tfidfFeatures = idfModel.transform(featureizedData)

    val lda = new LDA().setK(5).setSeed(123).setOptimizer("em").setFeaturesCol("features")
    val ldaModel = lda.fit(tfidfFeatures)
    val ldaTopics = ldaModel.describeTopics()
    ldaTopics.explain()
    ldaTopics.show(5)

    def indiceToWord = udf { indices: Seq[Int] => indices.map(vocabBroadcast.value.lift(_).getOrElse("UNK")) }
      //udf((indices: Seq[Int]) => indices.map(vocabBroadcast.value.lift(_).getOrElse("UNK")))
    val udf_mapping = udf(indiceToWord, ArrayType(StringType))
    val mapped_topics = ldaTopics.withColumn("termIndices", indiceToWord(col("termIndices")))
    mapped_topics.show(5, false)
    val ldaResults = ldaModel.transform(tfidfFeatures)
    ldaResults.show()
  }

  def stepThree(muellerData: DataFrame): Unit = {
    val muellerPipeline = customPipeline()
    val muellerModel = muellerPipeline.fit(muellerData) // Fit is not really doing anything as all pieces are pretrained
    val lightMuellerModel = new LightPipeline(muellerModel)
    val nlpPipelinePrediction: DataFrame = lightMuellerModel.transform(muellerData)

    nlpPipelinePrediction.select(explode($"ner_chunked.result").as("ner_chunks"))
      .groupBy("ner_chunks")
      .count
      .orderBy($"count".desc)
      .show(100, truncate = false)

    nlpPipelinePrediction.select($"ner_chunked.result".as("ner_chunks"))
      .show(100)

    ??? // Add your own investigation into data
  }
  def customPipeline(): Pipeline = {
    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    val sentence = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")
      .setExplodeSentences(true)
    val token = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")
    val normalized = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")
    val pos = PerceptronModel.pretrained()
      .setInputCols("sentence", "normalized")
      .setOutputCol("pos")
    val chunker = new Chunker()
      .setInputCols(Array("document", "pos"))
      .setOutputCol("pos_chunked")
      .setRegexParsers(Array(
        "<DT>?<JJ>*<NN>"
      ))
    val embeddings = WordEmbeddingsModel
      .pretrained()
      .setOutputCol("embeddings")
    val ner = NerDLModel.pretrained()
      .setInputCols("document", "normalized", "embeddings")
      .setOutputCol("ner")
    val nerConverter = new NerConverter()   // We want to chunk NER to create co-occurences.
      .setInputCols("document", "token", "ner")
      .setOutputCol("ner_chunked")
    val pipeline = new Pipeline().setStages(Array(
      document,
      sentence,
      token,
      normalized,
      pos,
      chunker,
      embeddings,
      ner,
      nerConverter
    ))

    pipeline
  }
}