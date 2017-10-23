import os
import json

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


DEFAULT = json.loads(os.environ.get('classifier', '{"type": "linearsvm"}'))

CLASSIFIERS = {
    "linearsvm": LinearSVC,
    "linearsvc": LinearSVC,
    "svm": SVC,
    "mlpclassifier": MLPClassifier,
    "ann": MLPClassifier,
    "mlp": MLPClassifier,
    "knn": KNeighborsClassifier,
    "sgd": SGDClassifier,
    "gnb": GaussianNB,
    "mnb": MultinomialNB
}


class ClassifierWrapper(object):
    def __init__(self, classifier=None, **kwargs):
        if classifier is None:
            d = DEFAULT.copy()
            classtype = d.pop('type', 'linearsvc')
            classtype = CLASSIFIERS[classtype]
            self.svc = classtype(**d)
        else:
            self.svc = classifier(**kwargs)

    def fit(self, X, y):
        # X = corpus2csc(X).T
        # self.num_terms = X.shape[1]
        self.svc.fit(X, y)
        return self

    def decision_function(self, Xnew):
        # Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        if hasattr(self.svc, 'decision_function'):
            return self.svc.decision_function(Xnew)
        else:
            return self.svc.predict_proba(Xnew)

    def predict(self, Xnew):
        # Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        ynew = self.svc.predict(Xnew)
        return ynew
