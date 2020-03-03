package help

import com.github.chen0040.lda.Lda
import com.londogard.smile.SmileOperators
import com.londogard.smile.extensions.*
import smile.clustering.KMeans
import smile.nlp.dictionary.EnglishStopWords
import smile.nlp.stemmer.PorterStemmer
import smile.stat.distribution.MultivariateGaussianMixture
import java.io.File

object LocalApproachFinished : SmileOperators {
    val simpleStemmer by lazy { PorterStemmer() }

    private fun readMuellerReport(): String = this::class.java.getResourceAsStream("/report_mueller.txt")
        .bufferedReader()
        .useLines { lines ->
            lines.map { it.normalize() }
                .filter { !it.isBlank() }
                .joinToString("\n")
        }

    fun <T> Iterator<T>.toList(): List<T> =
        ArrayList<T>().apply {
            while (hasNext())
                this += next()
        }

    @JvmStatic
    fun main(args: Array<String>) {
        val muellerReport = readMuellerReport()
        println("KeyWords")
        println(muellerReport.keywords(5))
        println("LDA")
        val lda = Lda().apply {
            topicCount = 5 // Play with this number
            maxVocabularySize = 20_000
            //isStemmerEnabled = true
            isRemoveNumber = true
            addStopWords(EnglishStopWords.DEFAULT.iterator().toList())
        }
        val ldaResult = lda.fit(muellerReport.split('\n'))

        println("Topic Count: ${ldaResult.topicCount()}")

        (0 until ldaResult.topicCount())
            .forEach { i ->
                val topicSummary = ldaResult.topicSummary(i)
                val topKeywords = ldaResult.topKeyWords(i, 10)
                val topStrings = ldaResult.topDocuments(i, 5)

                println("Topic #$i: $topicSummary")
                println(topKeywords.joinToString(prefix = "Keywords: ") { entry -> "${entry._1()} (${entry._2()})" })
                println(topStrings.joinToString(prefix = "Top Strings: ") { entry -> "Doc: (${entry._1().docIndex}, ${entry._2()}): ${entry._1().content}" })
                println("--")
            }
        ldaResult
            .documents()
            .shuffled()
            .take(5)
            .map { doc ->
                doc
                    .topTopics(3)
                    .joinToString(
                        separator = "\t",
                        prefix ="${doc.content}\nTop Topics: ",
                        postfix = "\n"
                    ) { "${it._1()} (score: ${it._2()})" }
            }
            .forEach(::println)
    }
}