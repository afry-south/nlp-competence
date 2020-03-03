import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn import metrics
from sklearn.svm import LinearSVC
from competitions.tools.preprocessing import PreProcessor
from tqdm import tqdm

tqdm.pandas()


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self._best_C, self._best_score, self._clf = None, None, None

    def predict(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_clf'])
        return self._clf.predict(X)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        self._clf = LinearSVC(C=self.C).fit(X, y)
        return self

    def train(self, X_train, y_train, X_val, y_val, Cs=None):
        """
        trainer to score auc over a grid of Cs
        Parameters
        ----------
        X_train, y_train, X_val, y_val: features and targets
        Cs: list of floats | int
        Return
        ------
        self
        """
        # init grid
        origin_C = self.C
        if Cs is None:
            Cs = [0.01, 0.1, 0.5, 1, 2, 10]
        # score
        scores = {}
        f1 = {}
        for C in Cs:
            # fit
            self.C = C
            model = self.fit(X_train, y_train)
            # predict
            y_pred = model.predict(X_val)
            scores[C] = metrics.roc_auc_score(y_val, y_pred)
            f1[C] = metrics.f1_score(y_val, y_pred)
            print("Val AUC Score: {:.4f}, F1: {:.4f} with C = {}".format(scores[C], f1[C], C))  # noqa

        # get max
        self._best_C, self._best_score = max(f1.items(), key=operator.itemgetter(1))  # noqa
        # reset
        self.C = origin_C
        return self

    @property
    def best_param(self):
        check_is_fitted(self, ['_clf'])
        return self._best_C

    @property
    def best_score(self):
        check_is_fitted(self, ['_clf'])
        return self._best_score


def transform(df_text):
    df_text.progress_apply(clean_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 strip_accents='ascii')
    return vectorizer.fit_transform(list(df_text))


def get_model():
    return BaseClassifier()


def clean_text(text):
    return PreProcessor(text).clean_and_get_text()

# Accuracy on the Quora dataset: 95.49
# F1 on the Quora dataset: 60.83# Now only 0.57 approx.. Why??
# TODO try to achieve above somehow
# TODO rebuild system to be fully CLASS!

# TODO
# char_transformer (3 chars a time)
# sparse.hstack([word_transformer(df_text), char_transformer(df_text)]).tocsr()
