from sklearn.svm import LinearSVC


class ClassifierWrapper(object):
    def __init__(self, classifier=LinearSVC):
        self.svc = classifier()
        self.num_terms = -1

    def fit(self, X, y):
        X = corpus2csc(X).T
        self.num_terms = X.shape[1]
        self.svc.fit(X, y)
        return self

    def decision_function(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        return self.svc.decision_function(Xnew)

    def predict(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        ynew = self.svc.predict(Xnew)
        return ynew
