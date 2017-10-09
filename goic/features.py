import re
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage.filters import gabor_kernel
import numpy as np
from .params import OPTION_DELETE, OPTION_GROUP, OPTION_NONE



class Features:
    FREQ = [0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]
    THETA = [0, np.pi / 4, np.pi / 3, np.pi/2, np.pi/1.5, np.pi/1.2, p.pi]

    def __init__(self, docs, gabor, resize=(270, 270), **kwargs):
        self.gabor = gabor
        self.resize = resize

    def __getitem__(self, filename):
        img = imread(filename, as_grey=True)
        img = resize(img, self.size, mode='edge')
        gimg = None
        for g in self.gabor:
            freq, theta = g
            x = gabor_kernel(Features.FREQ[freq], theta=Features.THETA[theta])
            if gimg is None:
                gimg = x.real
            else:
                gimg = gimg + x.real

        vec = hog(gimg)
        print(filename, vec, file=sys.stderr)
        return vec
