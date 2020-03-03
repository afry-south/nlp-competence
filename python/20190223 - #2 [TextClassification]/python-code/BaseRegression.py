import regex as re
import operator
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import tqdm

tqdm.pandas()


class BaseRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=True, n_jobs=-1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(X.multiply(self._r))

    def predict_proba(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(X.multiply(self._r))[:, 1]

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        def pr(X, y_i, y):
            p = X[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(X, 1, y) / pr(X, 0, y)))
        X_nb = X.multiply(self._r)
        self._clf = LogisticRegression(
            C=self.C,
            dual=self.dual,
            # multi_class='ovr',
            n_jobs=self.n_jobs
        ).fit(X_nb, y)
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
            y_proba = model.predict(X_val)
            scores[C] = metrics.roc_auc_score(y_val, y_proba)
            f1[C] = metrics.f1_score(y_val, y_proba)
            print("Val AUC Score: {:.4f}, F1: {:.4f} with C = {}".format(scores[C], f1[C], C))  # noqa

        # get max
        self._best_C, self._best_score = max(scores.items(), key=operator.itemgetter(1))  # noqa
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
                                 lowercase=True,
                                 strip_accents='ascii')
    return vectorizer.fit_transform(list(df_text))


def get_model():
    return BaseRegression()


def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing punctuation
    - Lowering text
    """

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)  # Extra: Is regex needed? Other ways to accomplish this.
    text = re.sub(r"\"", "", text)
    # replace all non alphanumeric with space
    text = re.sub(r"\W+", " ", text)
    # text = re.sub(r"<.+?>", " ", text) # <br></br>hej<br></br>
    return text.strip().lower()
