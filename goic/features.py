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
import math
import sys
from scipy.stats import entropy


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
    # feats = np.zeros((len(kernels), 2), dtype=np.double)
    results = []
    for k, kernel in enumerate(kernels):
        filtered = filter.convolve(image, kernel, mode='wrap')
        results.append(filtered)
 
    return results


def get_vector(obj, path_file):
    #Tomo los indices de la configuracion
    indices = convertir_bi_uni(obj.gabor)
    k = kernels_subset(obj.kernels, indices)
    # imagen = io.imread(path_file, as_grey=True)
    imagen = io.imread(path_file)
    imagen = resize(imagen, obj.resize, mode='edge')

    if obj.contrast == 'sub-mean':
        for i in range(3):
            imagen[:,:,i] = imagen[:,:,i] - imagen[:,:,i].mean()

    imagen = rgb2gray(imagen)

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
    if(len(obj.gabor) == 0):
        sumG = img
    elif(len(obj.gabor) > 1):
        sumG = suma_imagenes(GG)
    else:
        sumG = GG[0]
    # Una vez calculadas las imagenes sacamos el HOG
    # vec = hog(sumG, orientations=8, pixels_per_cell=obj.pixels_per_cell, cells_per_block=obj.cells_per_block, block_norm='L2-#Hys')
    if obj.vector == 'hog':
        orientations = 8
        vec = hog(sumG, orientations=orientations, pixels_per_cell=obj.pixels_per_cell, cells_per_block=obj.cells_per_block, block_norm='L1')
        # vec = daisy(sumG, step=64, radius=32, rings=3).flatten()
        # if obj.vector == 'pi-hog':
        #    m = orientations * obj.cells_per_block[0] * obj.cells_per_block[#1]
        #    XX = np.split(vec, len(vec) // m)
        # X = [(float(entropy(x)), x) for x in XX]
        # X = [(np.random.rand(), x) for x in XX]
        # print(X[:4])
        # X.sort()
        # return np.concatenate([np.random.rand(m) for x in XX])
        # return np.concatenate([x[-1] for x in X])
    else:
        raise Exception("Unknown feature detection {0}".format(obj.vector))

    return vec


def suma_imagenes(lista_imgs):
    size = lista_imgs[0].shape
    r = np.zeros(size)

    for x in lista_imgs:
        r += x

    return(r)


class Features:
    def __init__(self, docs, gabor, resize=(270, 270), equalize=False, edges='none', pixels_per_cell=(32, 32), cells_per_block=(3,3), contrast='none', vector='hog', **kwargs):
        self.gabor = gabor
        self.resize = resize
        self.kernels = generacion_kernels()
        self.equalize = equalize
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.edges = edges
        self.contrast = contrast
        self.vector = vector

    def __getitem__(self, filename):
        # print("==== processing", filename, ", gabor: ", self.gabor, ", resize: ",  self.resize, file=sys.stderr)
        x = get_vector(self, filename)
        # print(len(x), file=sys.stderr)
        return x

