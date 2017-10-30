# author: Eric S. Tellez <eric.tellez@infotec.mx>

import os
import json
import numpy as np
from numpy.random import randint

import logging
from itertools import combinations


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


class Fixed:
    def __init__(self, value):
        self.value = value
        self.valid_values = [value]

    def neighborhood(self, v):
        return []

    def get_random(self):
        return self.value


class SetVariable:
    def __init__(self, values):
        self.valid_values = list(values)

    def neighborhood(self, value):
        return [u for u in self.valid_values if u != value]

    def get_random(self):
        i = randint(len(self.valid_values))
        return self.valid_values[i]


class PowersetVariable:
    def __init__(self, initial_set, max_size=None):
        self.valid_values = []
        if max_size is None:
            max_size = len(initial_set) // 2 + 1

        for i in range(1, len(initial_set)+1):
            for l in combinations(initial_set, i):
                if len(l) <= max_size:
                    self.valid_values.append(l)

    def mismatches(self, value):
        lvalue = len(value)
        for v in self.valid_values:
            # if len(value.intersection(v)) == lvalue - 1 or len(value.union(v)) == lvalue + 1:
            ulen = len(value.union(v))
            ilen = len(value.intersection(v))
            if ulen in (lvalue, lvalue + 1) and ilen in (lvalue, lvalue - 1):
                yield v

    def neighborhood(self, value):
        L = []
        for v in value:
            if isinstance(v, list):
                v = tuple(v)
            L.append(v)

        return list(self.mismatches(set(L)))

    def get_random(self):
        x = len(self.valid_values)
        i = randint(x)
        return self.valid_values[i]


class PowerGridVariable:
    def __init__(self, max_enabled, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.max_enabled = max_enabled

    def get_random(self):
        L = []
        m = randint(1, self.max_enabled+1)
        # if m == 0:
        #     return [(0, 0)]

        for i in range(m):
            L.append((randint(0, self.nrows),
                      randint(0, self.ncols)))

        return list(set(L))

    def neighborhood(self, value):
        for i in range(len(value)):
            row, col = value[i]
            if row - 1 >= 0:
                conf = list(value)
                conf[i] = (row - 1, col)
                yield sorted(set(conf))
            if row + 1 < self.nrows:
                conf = list(value)
                conf[i] = (row + 1, col)
                yield sorted(set(conf))
            if col - 1 >= 0:
                conf = list(value)
                conf[i] = (row, col - 1)
                yield sorted(set(conf))
            if col + 1 < self.ncols:
                conf = list(value)
                conf[i] = (row, col + 1)
                yield sorted(set(conf))

            #if len(value) > 1:
            conf = list(value)
            conf.pop(i)
            yield conf


OPTION_NONE = 'none'
OPTION_GROUP = 'group'
OPTION_DELETE = 'delete'
BASIC_OPTIONS = [OPTION_DELETE, OPTION_GROUP, OPTION_NONE]


def Option():
    return SetVariable(BASIC_OPTIONS)


def Uniform(left, right, k=10):
    d = (right - left) * np.random.random_sample(k) + left
    return SetVariable(d)


def Normal(mean, sigma, k=10):
    d = mean * np.random.randn(k) + sigma
    return SetVariable(d)


def Boolean():
    return SetVariable([False, True])


DefaultParams = {
    "gabor": PowerGridVariable(7, 5, 8),
    # "gabor": Fixed([]),
    # "resize": SetVariable([(320, 320), (380, 380), (270, 270), (420, 420)]),
    # "resize": Fixed((320, 320)),
    "resize": Fixed((225, 225)),
    "equalize": SetVariable(['none', 'local:10', 'global']),
    # "edges": SetVariable(['none', 'scharr', 'sobel', 'prewitt', 'roberts']),
    "edges": SetVariable(['none', 'scharr', 'sobel']),
    "contrast": SetVariable(['none', 'sub-mean']),
    #"pixels_per_cell": SetVariable([(32, 32), (64, 64), (128, 128)]),
    "pixels_per_cell": SetVariable([(16, 16), (24, 24), (32, 32)]),
    # "pixels_per_cell": Fixed((32, 32)),
    "cells_per_block": SetVariable([(2, 2), (3, 3)]),
    # "cells_per_block": Fixed((3, 3)),
    # "vector": SetVariable(["hog", "pi-hog", "orb", "hog-orb", "lbp-hog"])
    "vector": Fixed("hog"),
    # "vector": Fixed("lbp-hog"),
    # "channels": SetVariable(["green", "rgb"]),
    "channels": Fixed("rgb"),
    # "correlation":SetVariable(["none", "yes"])
    "correlation": Fixed(False)  # "/home/daniela/GOIC/GOIC/mascara.png"
}


for key, value in json.loads(os.environ.get("params", '{}')).items():
    DefaultParams[key] = Fixed(value)


class ParameterSelection:
    def __init__(self, params=None):
        if params is None:
            params = DefaultParams

        self.params = params

    def sample_param_space(self, n):
        for i in range(n):
            kwargs = {}
            for k, v in self.params.items():
                kwargs[k] = v.get_random()

            yield kwargs

    def expand_neighbors(self, s, keywords=None):
        if keywords is None:
            keywords = set(s.keys())

        for k, v in sorted(s.items()):
            if k[0] == '_' or k not in keywords:
                # by convention, metadata starts with underscore
                continue

            vtype = self.params[k]
            if isinstance(vtype, Fixed):
                continue

            for neighbor in vtype.neighborhood(v):
                x = s.copy()
                x[k] = neighbor
                yield(x)

    def get_best(self, fun_score, cand, desc="searching for params", pool=None):
        if pool is None:
            # X = list(map(fun_score, cand))
            X = [fun_score(x) for x in tqdm(cand, desc=desc, total=len(cand))]
        else:
            # X = list(pool.map(fun_score, cand))
            X = [x for x in tqdm(pool.imap_unordered(fun_score, cand), desc=desc, total=len(cand))]

        # a list of tuples (score, conf)
        X.sort(key=lambda x: x['_score'], reverse=True)
        return X

    def search(self, fun_score, bsize=32, hill_climbing=True, pool=None, best_list=None):
        # initial approximation, montecarlo based procesess

        tabu = set()  # memory for tabu search
        if best_list is None:
            L = []
            for conf in self.sample_param_space(bsize):
                code = _identifier(conf)
                if code in tabu:
                    continue

                tabu.add(code)
                L.append((conf, code))

            best_list = self.get_best(fun_score, L, pool=pool)
        else:
            for conf in best_list:
                tabu.add(_identifier(conf))

        def _hill_climbing(keywords, desc):
            # second approximation, a hill climbing process
            i = 0
            while True:
                i += 1
                bscore = best_list[0]['_score']
                L = []

                for conf in self.expand_neighbors(best_list[0], keywords=keywords):
                    code = _identifier(conf)
                    if code in tabu:
                        continue

                    tabu.add(code)
                    L.append((conf, code))

                best_list.extend(self.get_best(fun_score, L, desc=desc + " {0}".format(i), pool=pool))
                best_list.sort(key=lambda x: x['_score'], reverse=True)
                if bscore == best_list[0]['_score']:
                    break

        if hill_climbing:
            _hill_climbing(None, "hill climbing optimization")

        return best_list


def _identifier(conf):
    return ",".join(map(str, conf.items()))
