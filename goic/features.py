import numpy as np
from skimage.util import img_as_float
from skimage.filters import gabor_kernel, sobel, scharr, prewitt, roberts, rank
from skimage.morphology import disk
from skimage.transform import resize
from skimage import data, io, filters
from scipy import ndimage as ndi
import scipy.ndimage.filters as filter
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from numba import jit
from numpy import arange
import math
import sys

@jit
def convertir_bi_uni(lista_tupla):
    indices = []
    for i in lista_tupla:
        indice = (i[0] * 8 + i[1])
        indices.append(indice)

    return indices


def kernels_subset(kernels, indices):
    kernels_copy = []
    for i in indices:
        kernels_copy.append(kernels[i])
    return(kernels_copy)


def generacion_kernels():
    kernels = []  # Aquí se guardan los filtros que vamos a generar
    # thetas = frange(0,((7*np.pi)/8),pi/8)
    theta = 0
    while theta <= 7 * np.pi / 8:
        # for theta in thetas:
        sigma = 2*np.pi
        for frequency in (0.10, 0.15, 0.20, 0.25, 0.35):
            kernel = np.real(gabor_kernel(frequency, theta=theta,sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
        theta += np.pi / 8
    return kernels


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    results = []
    for k, kernel in enumerate(kernels):
        filtered = filter.convolve(image, kernel, mode='wrap')
        results.append(filtered)
    return results

@jit
def get_vector(lista_config, path_file, obj):
    #Tomo los indices de la configuracion
    indices = convertir_bi_uni(lista_config)
    k = kernels_subset(obj.kernels, indices)
    imagen = io.imread(path_file, as_grey=True)
    imagen = resize(imagen, obj.resize, mode='edge')

    if obj.equalize != 'none':
        if obj.equalize == 'global':
            imagen = exposure.equalize_hist(imagen)
        else:
            d = int(obj.equalize.split(':')[-1] or 30)
            imagen = rank.equalize(imagen, selem=disk(d))

    if obj.edges != 'none':
        if obj.edges == 'sobel':
            imagen = sobel(imagen)
        elif obj.edges == 'scharr':
            imagen = scharr(imagen)
        elif obj.edges == 'prewitt':
            imagen = prewitt(imagen)
        elif obj.edges == 'roberts':
            imagen = roberts(imagen)
        else:
            raise ArgumentException("Unknown edge detector {0}".format(obj.edges))

    img = img_as_float(imagen)
    GG = []
    #Se calcula la Gabor image para cada filtro especificado
    GG = compute_feats(img, k)
    if(len(lista_config) == 0):
        sumG = img
    elif(len(lista_config) > 1):
        sumG = suma_imagenes(GG)
    else:
        sumG = GG[0]
    #Una vez calculadas las imagenes sacamos el HOG
    h_Gabor = hog(sumG, orientations=8, pixels_per_cell=obj.pixels_per_cell, cells_per_block=obj.cells_per_block, block_norm='L2-Hys')
    return(h_Gabor)

@jit
def suma_imagenes(lista_imgs):

    size = lista_imgs[0].shape
    r = np.zeros(size)

    for x in lista_imgs:
        r += x

    return(r)

class Features:
    def __init__(self, docs, gabor, resize=(270, 270), equalize=False, edges='none', pixels_per_cell=(32, 32), cells_per_block=(3,3), **kwargs):
        self.gabor = gabor
        self.resize = resize
        self.kernels = generacion_kernels()
        self.equalize = equalize
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.edges = edges

    def __getitem__(self, filename):
        #img = imread(filename, as_grey=True)
        #img = resize(img, self.size, mode='edge')
        # print("==== processing", filename, ", gabor: ", self.gabor, ", resize: ",  self.resize, file=sys.stderr)

        vec = get_vector(self.gabor, filename, self)

        return(vec)
