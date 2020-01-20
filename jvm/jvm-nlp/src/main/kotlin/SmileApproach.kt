import com.londogard.smile.SmileOperators
import com.londogard.smile.extensions.bag
import com.londogard.smile.extensions.keywords
import com.londogard.smile.extensions.normalize
import com.londogard.smile.extensions.words
import smile.clustering.KMeans
import smile.nlp.stemmer.PorterStemmer
import smile.stat.distribution.MultivariateGaussianMixture

object SmileApproach: SmileOperators {
    val simpleStemmer by lazy { PorterStemmer() }

    private fun readMuellerReport(): String = this::class.java.getResourceAsStream("report_mueller.txt")
        .bufferedReader()
        .useLines { lines ->
            TODO("Implement how to modify lines") // normalize, remove stop-words etc
        }

    @JvmStatic
    fun main(args: Array<String>) {
        val muellerReport = readMuellerReport()

        val corpus = muellerReport.split('\n').map { it.bag() }
        val features = muellerReport.bag().entries.asSequence().sortedByDescending { it.value }.take(512).map { it.key }.toList()
        val bags = corpus.map { vectorize(features, it) }.filter { it.find { it > 0 } != null }
        val x = tfidf(bags).map { it.toDoubleArray() }.toTypedArray()
        MultivariateGaussianMixture(x, 10)//components/.mapIndexed { i, itm =>  }
        println(muellerReport.keywords(5))
        //val kmeans = KMeans(x, 10, 100, 20)
        //println(kmeans)
    }
}