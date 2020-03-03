from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class Base(BaseEstimator, ClassifierMixin):
    @property
    def best_param(self):
        check_is_fitted(self, ['_clf'])
        return self._best_C

    @property
    def best_score(self):
        check_is_fitted(self, ['_clf'])
        return self._best_score
