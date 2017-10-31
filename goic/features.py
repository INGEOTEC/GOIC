import numpy as np
from skimage.util import img_as_float
from skimage.filters import gabor_kernel, sobel, scharr, prewitt, roberts, rank
from skimage.morphology import disk
from skimage.transform import resize
from skimage import data, io, filters
from scipy import ndimage as ndi
import scipy.ndimage.filters as filter
from skimage.color import rgb2gray
from skimage.feature import hog, daisy
from skimage import exposure
from numpy import arange
import skimage
import math
import sys
from scipy.stats import entropy
from skimage.feature import ORB, match_descriptors, local_binary_pattern
from skimage.feature import match_template
from sklearn.cluster import KMeans

class Features:
    def __init__(self,
                 docs,
                 resize=(225, 225),
                 equalize=False,
                 edges='none',
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3),
                 contrast='none',
                 features='hog-vow',
                 channels='rgb',
                 correlation='yes',
                 sample_size=50000,
                 num_centers=223,
                 encoding='hist',
                 **kwargs):

        self.resize = resize
        self.equalize = equalize
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.edges = edges
        self.contrast = contrast
        self.features = features
        self.channels = channels
        self.correlation = correlation
        self.encoding = encoding
        self.train_features = {}  # a hack to avoid the training double processing of the whole dataset
        vectors = []
        for filename in docs:
            V = []
            for vec in self.compute_features(filename):
                V.append(vec)
                vectors.append(vec)

            self.train_features[filename] = V
        
        print("the training set contains {0} vectors".format(len(vectors)), file=sys.stderr)
        if len(vectors) > sample_size:
            np.random.shuffle(vectors)
            vectors = vectors[:sample_size]
            print("we kept {0} vectors, randomly selected".format(len(vectors)), file=sys.stderr)

        self.model = KMeans(num_centers, verbose=1)

        print("preparing to fit our codebook with {0} centers".format(num_centers), file=sys.stderr)
        self.model.fit(vectors)
        vectors = None

        print("encoding our training vectors", file=sys.stderr)
        for filename, veclist in self.train_features.items():
            self.train_features[filename] = self.encode(veclist)

        print("our feature module is fitted", file=sys.stderr)
    
    def encode(self, veclist):
        if self.encoding == 'hist':
            return self.hist(veclist)
        elif self.encoding == 'seq':
            return self.sequence(veclist)
        else:
            raise Exception("Unknown feature encoding {0}".format(self.encoding))

    def sequence(self, veclist):
        seq = []
        for vec in veclist:
            c = np.argmin(self.model.transform(vec))
            seq.append(c)

        return np.array(seq, dtype=np.int32)

    def hist(self, veclist):
        h = np.zeros(self.model.n_clusters)
        for vec in veclist:
            c = np.argmin(self.model.transform(vec))
            h[c] += 1

        return h

    def __getitem__(self, filename):
        # print("==== processing", filename, ", gabor: ", self.gabor, ", resize: ",  self.resize, file=sys.stderr)
        if self.train_features:
            s = self.train_features.get(filename, None)
            if s:
                return s
    
            self.train_features = None  # if we reach this code we are beyond the training phase
    
        return self.encode(self.compute_features(filename))

    def compute_features(self, path_file):
        imagen = io.imread(path_file)
        imagen = resize(imagen, self.resize, mode='edge')

        if self.contrast == 'sub-mean':
            for i in range(3):
                imagen[:, :, i] = imagen[:, :, i] - imagen[:, :, i].mean()

        if self.channels == "red":
            imagen = imagen[:, :, 0]
        elif self.channels == "green":
            imagen = imagen[:, :, 1]
        elif self.channels == "blue":
            imagen = imagen[:, :, 2]
        else:
            imagen = rgb2gray(imagen)

        if self.equalize != 'none':
            if self.equalize == 'global':
                imagen = exposure.equalize_hist(imagen)
            else:
                d = int(self.equalize.split(':')[-1])
                imagen = rank.equalize(imagen, selem=disk(d))

        if self.edges != 'none':
            if self.edges == 'sobel':
                imagen = sobel(imagen)
            elif self.edges == 'scharr':
                imagen = scharr(imagen)
            elif self.edges == 'prewitt':
                imagen = prewitt(imagen)
            elif self.edges == 'roberts':
                imagen = roberts(imagen)
            else:
                raise Exception("Unknown edge detector {0}".format(self.edges))

        if self.correlation:
            mascara = io.imread(self.correlation)
            mascara = rgb2gray(mascara)
            mascara = np.array(mascara)
            mascara = skimage.transform.resize(mascara, (25, 25), mode='edge')
            resultado = match_template(imagen, mascara)
            resultado = resultado + abs(resultado.min())
            resultado[~((resultado < 0.2) | (resultado > resultado.max()-0.2))] = 0.6
            img = resultado
        else:
            img = img_as_float(imagen)

        if self.features.startswith('hog'):
            orientations = 8
            vec = hog(img, orientations=orientations, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, block_norm='L2-Hys')

            if self.features == 'hog':
                return [vec]
            elif self.features == 'hog-bovw':
                m = orientations * self.cells_per_block[0] * self.cells_per_block[1]
                XX = np.split(vec, len(vec) // m)
                return XX
            else:
                raise Exception("Unknown feature detection {0}".format(self.features))

        elif self.features == 'daisy':
            return daisy(img, step=32, radius=16, rings=3).flatten()
        else:
            raise Exception("Unknown feature detection {0}".format(self.features))
