import com.github.chen0040.lda.Lda
import com.github.chen0040.lda.LdaResult
import com.londogard.smile.SmileOperators
import com.londogard.smile.extensions.*
import smile.clustering.KMeans
import smile.nlp.dictionary.EnglishStopWords
import smile.nlp.stemmer.PorterStemmer
import smile.stat.distribution.MultivariateGaussianMixture

object LocalApproach: SmileOperators {
    val simpleStemmer by lazy { PorterStemmer() }
    val englishStopWords by lazy { EnglishStopWords.DEFAULT.iterator().toList() }

    private fun readMuellerReport(): String = this::class.java.getResourceAsStream("report_mueller.txt")
        .bufferedReader()
        .useLines { lines ->
            TODO("Implement how to modify lines, perhaps normalizing, removing stopWords etc")
        }

    fun <T> Iterator<T>.toList(): List<T> =
        TODO("Create an ArrayList<T> and apply to add each element from iterator")

    @JvmStatic
    fun main(args: Array<String>) {
        val muellerReport = readMuellerReport()
        println("Keywords: ${muellerReport.keywords(5)}")
        println("LDA")
        val lda = Lda().apply {
            TODO("Set a topic count, max_vocab, stop_words etc - play around")
        }
        val ldaResult: LdaResult = TODO("Fit LDA to the data")
        println("Topic Count: ${ldaResult.topicCount()}")
        TODO("For each topic print summary, topKeyWords & topDocuments (extra: make use of joinToString)")
        TODO("For each document print the content")
    }
}