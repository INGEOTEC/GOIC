# Copyright 2016 Mario Graff (https://github.com/mgraffg)
# Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import sys
from time import time

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn import preprocessing
from sklearn import cross_validation
from .classifier import ClassifierWrapper
from .features import Features


class ScoreSampleWrapper(object):
    def __init__(self, X, y, Xstatic=[], ystatic=[], ratio=0.8, test_ratio=None, score='macrof1', classifier=ClassifierWrapper, random_state=None):
        assert ratio < 1, "ratio {0} is invalid, valid values are 0 < ratio < 1".format(ratio)
        self.score = score
        self.le = preprocessing.LabelEncoder().fit(y)
        self.create_classifier = classifier
        if test_ratio is None:
            test_ratio = 1.0 - ratio

        I = list(range(len(y)))
        np.random.shuffle(I)
        s = int(np.ceil(len(y) * ratio))
        s_end = int(np.ceil(len(y) * test_ratio))
        y = self.le.transform(y)
        train, test = I[:s], I[s:s+s_end]
        self.train_corpus = [X[i] for i in train]
        self.train_corpus.extend(Xstatic)

        if len(ystatic) > 0:
            ystatic = self.le.transform(ystatic)
            self.train_y = np.hstack((y[train], ystatic))
        else:
            self.train_y = y[train]

        self.test_corpus = [X[i] for i in test]
        self.test_y = y[test]

    def __call__(self, args):
        # ("testing SampleWrapper: ", args, file=sys.stderr)

        if isinstance(args, (tuple, list)):
            conf, code = args
            create_classifier = self.create_classifier
            model = None
        else:
            conf = args.copy()
            classifier = args.pop("type")
            model = lambda item: np.array(item)
            create_classifier = lambda: classifier(**args)
    
        st = time()

        if model is None:
            model_klass = os.environ.get("klasses", None)

            if model_klass:
                model_klass = self.le.transform(model_klass.split(','))
                _train = [self.train_corpus[i] for i in len(self.train_corpus) if self.train_y[i] in model_klass]
                model = Features(_train, **conf)
            else:
                model = Features(self.train_corpus, **conf)
            
            train_X = [model[doc] for doc in self.train_corpus]
            test_X = [model[doc] for doc in self.test_corpus]
        else:
            train_X = [model(doc) for doc in self.train_corpus]
            test_X = [model(doc) for doc in self.test_corpus]

        c = create_classifier()
        c.fit(train_X, self.train_y)

        pred_y = c.predict(test_X)
        self.compute_score(conf, pred_y)
        conf['_time'] = (time() - st)
        return conf

    def compute_score(self, conf, hy):
        RS = recall_score(self.test_y, hy, average=None)
        conf['_all_f1'] = M = {str(self.le.inverse_transform([klass])[0]): f1 for klass, f1 in enumerate(f1_score(self.test_y, hy, average=None))}
        conf['_all_recall'] = {str(self.le.inverse_transform([klass])[0]): r for klass, r in enumerate(RS)}
        conf['_all_precision'] = {str(self.le.inverse_transform([klass])[0]): f1 for klass, f1 in enumerate(precision_score(self.test_y, hy, average=None))}
        conf['_macrorecall'] = np.mean(RS)

        if len(self.le.classes_) == 2:
            conf['_macrof1'] = np.mean(np.array([v for v in conf['_all_f1'].values()]))
            conf['_weightedf1'] = conf['_microf1'] = f1_score(self.test_y, hy, average='binary')
        else:
            conf['_macrof1'] = f1_score(self.test_y, hy, average='macro')
            conf['_microf1'] = f1_score(self.test_y, hy, average='micro')
            conf['_weightedf1'] = f1_score(self.test_y, hy, average='weighted')

        # conf['_macroauc'] = roc_auc_score(self.test_y, hy, average='macro')
        # conf['_microauc'] = roc_auc_score(self.test_y, hy, average='micro')

        m = 1.0
        for x in conf['_all_f1'].values():
            m *= x + 0.0001

        conf['_geometricf1'] = m
        # the extra 0.0001 is introduced to avoid divisions by zero
        conf['_harmonicf1'] = len(conf['_all_f1']) / sum([1/(x+0.0001) for x in conf['_all_f1'].values()])

        conf['_accuracy'] = accuracy_score(self.test_y, hy)
        conf['_macrof1accuracy'] = conf["_macrof1"] * conf["_accuracy"]

        if self.score.startswith('avgf1:'):
            klist = [M[x] for x in self.score.replace('avgf1:', '').split(':')]
            conf['_' + self.score] = sum(klist) / len(klist)

        conf['_score'] = conf['_' + self.score]


class ScoreKFoldWrapper(ScoreSampleWrapper):
    def __init__(self, X, y, Xstatic=[], ystatic=[], nfolds=5, score='macrof1', classifier=ClassifierWrapper, random_state=None):
        self.nfolds = nfolds
        self.score = score
        # self.X = np.array(X)
        self.X = X
        self.Xstatic = Xstatic
        self.le = preprocessing.LabelEncoder().fit(y)
        self.y = self.le.transform(y)
        if len(ystatic) > 0:
            self.ystatic = self.le.transform(ystatic)
        else:
            self.ystatic = []
        self.test_y = self.y
        self.create_classifier = classifier
        self.kfolds = cross_validation.StratifiedKFold(y, n_folds=nfolds, shuffle=True, random_state=random_state)

    def __call__(self, args):
        # print("testing kfolds: ", args, file=sys.stderr)
        if isinstance(args, (tuple, list)):
            conf, code = args
            create_classifier = self.create_classifier
            model = None
        else:
            conf = args.copy()
            classifier = args.pop("type")
            ckwargs = args
            model = lambda item: np.array(item)
            create_classifier = lambda: classifier(**args)

        st = time()
        predY = np.zeros(len(self.y))
        # X = np.array(self.X)
        for train, test in self.kfolds:
            # A = X[train]
            A = [self.X[i] for i in train]
            if len(self.Xstatic) > 0:
                A.extend(self.Xstatic)

            trainY = self.y[train]
            if len(self.ystatic) > 0:
                trainY = np.hstack((trainY, self.ystatic))

            # print(model, args, type(args), file=sys.stderr)
            if model is None:
                _model = Features(A, **conf)                
                # _model = Features([X[i] for i in train], **conf)
                trainX = [_model[x] for x in A]
                testX = [_model[self.X[i]] for i in test]
            else:
                trainX = [model(x) for x in A]
                testX = [model(self.X[i]) for i in test]

            c = create_classifier()
            try:
                c.fit(trainX, trainY)
            except ValueError:
                conf["_error"] = "this configuration produces an empty matrix"
                conf["_score"] = 0.0
                return conf

            predY[test] = c.predict(testX)

        self.compute_score(conf, predY)
        conf['_time'] = (time() - st) / self.nfolds
        return conf
