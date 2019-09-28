#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import scipy.io
from funciones import obtenerProbAPriori, obtenerMatricesMedias, obtenerMatricesCovarianzas
#%%
path_datos_x = glob.glob('./Dataset/Train/*data.tif')
#%%
datos_x  = np.array([np.dstack(np.array(rasterio.open(path).read())) for path in path_datos_x])
#%%
# máscaras
# Value / Interpretation
# 0    Shadow
# 1    Shadow over Water
# 2    Water
# 3    Snow
# 4    Land
# 5    Cloud
# 6    Flooded
path_datos_y = glob.glob('./Dataset/Train/*mask.mat')
datos_y = np.array([scipy.io.loadmat(path)['img'] for path in path_datos_y])
clases = {
    0 : 'Sombra',
    1 : 'Sombra sobre agua',
    2 : 'Agua',
    3 : 'Nieve',
    4 : 'Tierra',
    5 : 'Nube',
    #6 : 'Inundado',
}
n_clases = len(clases)
#%%
probabilidades = obtenerProbAPriori(datos_y, n_clases)

#%%
medias = obtenerMatricesMedias(datos_x, datos_y, clases, probabilidades)

#%%
#covarianzas = obtenerMatricesCovarianzas(datos_x, datos_y, medias, clases, probabilidades)