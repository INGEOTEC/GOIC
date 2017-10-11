import pandas as pd
import numpy as np
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.transform import resize
from skimage import data, io, filters
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import hog
from numba import jit
from numpy import arange
import sys


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
    #number = 10 para los 40
    kernels = []  # Aquí se guardan los filtros que vamos a generar
    for theta in range(10):
        # Theta irá iterando con incrementos de theta/4*pi (recordemos que thetha #son los grados en radianes)
        theta = theta / 4. * np.pi
        for sigma in (1, 3): #sigma variará de 1 a 3
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    results = []
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        results.append(filtered)
    return results

def get_vector(lista_config, path_file, kernels, size):
    #kernels = generacion_kernels()
    #Tomo los indices de la configuracion
    indices = convertir_bi_uni(lista_config)
    k = kernels_subset(kernels, indices)
    imagen = io.imread(path_file)
    imagen = resize(imagen, size, mode='edge')
    img_gray = rgb2gray(imagen)
    #rezise de la imagen?
    img = img_as_float(img_gray)
    GG = []
    #Se calcula la Gabor image para cada filtro especificado
    GG = compute_feats(img, k)
    #Una vez calculadas las imagenes sacamos el HOG
    h_Gabor = hog(GG[0], orientations=8, pixels_per_cell=(64, 64), cells_per_block=(2, 2), block_norm='L2-Hys')
    return(h_Gabor)

def get_vector_combinacion(lista_config, path_file, kernels, size):
    #kernels = generacion_kernels()
    #Tomo los indices de la configuracion
    indices = convertir_bi_uni(lista_config)
    k = kernels_subset(kernels, indices)
    imagen = io.imread(path_file)
    imagen = resize(imagen, size, mode='edge')
    img_gray = rgb2gray(imagen)
    #rezise de la imagen?
    img = img_as_float(img_gray)
    GG = []
    #Se calcula la Gabor image para cada filtro especificado
    GG = compute_feats(img, k)
    sumG = suma_imagenes(GG)
    #Una vez calculadas las imagenes sacamos el HOG
    h_Gabor = hog(sumG, orientations=8, pixels_per_cell=(64, 64), cells_per_block=(2, 2), block_norm='L2-Hys')
    print(type(h_Gabor), h_Gabor.shape, file=sys.stderr)
    return(h_Gabor)

@jit
def suma_imagenes(lista_imgs):
    d = len(lista_imgs)

    size = lista_imgs[0].shape
    r = np.zeros(size)

    for k in range(0,d):
        r += lista_imgs[k]

        # for i in range(0,size[0]):
        #     for j in range(0, size[1]):
        #         r[i][j] = r[i][j] + lista_imgs[k][i][j]
    return(r)

class Features:
    def __init__(self, docs, gabor, resize=(270, 270), **kwargs):
        self.gabor = gabor
        self.resize = resize
        self.kernels = generacion_kernels()

    def __getitem__(self, filename):
        #img = imread(filename, as_grey=True)
        #img = resize(img, self.size, mode='edge')
        print("==== processing", filename, ", gabor: ", self.gabor, ", resize: ",  self.resize, file=sys.stderr)
        if(len(self.gabor)) > 1:
            vec = get_vector_combinacion(self.gabor, filename, self.kernels, self.resize)
        else:
            vec = get_vector(self.gabor, filename, self.kernels, self.resize)

        return(vec)
