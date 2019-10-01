# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:36:54 2019

@author: Andres
"""

#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import glob

#%%
path_datos_x = glob.glob('./Dataset/Train/*data.tif')
datos_x  = np.array([np.dstack(np.array(rasterio.open(path).read())) for path in path_datos_x])
datos_x = datos_x[:,:,:,1:4]#/65536.0

path_datos_y = glob.glob('./Dataset/Train/*mask.png')
mask_train = np.array([plt.imread(path) for path in path_datos_y])

path_datos_prueba = glob.glob('./Dataset/Test/*data.tif')
datos_prueba  = np.array([np.dstack(np.array(rasterio.open(path).read())) for path in path_datos_prueba])
datos_prueba = datos_prueba[:,:,:,1:4]#/65536.0

path_datos_y_test = glob.glob('./Dataset/Test/*pred.png')
mask_test = np.array([plt.imread(path) for path in path_datos_y_test])
#%%
imgs_train = np.flip(datos_x, 3)
imgs_test = np.flip(datos_prueba, 3)

#%%
def norm(datos):
    return (datos-datos.min())/(datos.max()-datos.min())

imgs_train = norm(imgs_train)
imgs_test = norm(imgs_test)
#%%
# Value / Interpretation
# 0    Shadow
# 1    Shadow over Water
# 2    Water
# 3    Snow
# 4    Land
# 5    Cloud
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
sombra = np.array([0., 0., 0., 1.])
sombra_sa = np.array([0., 0., 0.5, 1.])
agua = np.array([0., 0., 1., 1.])
nieve = np.array([0., 1., 1., 1.])
tierra = np.array([0.5, 0.5, 0.5, 1.])
nube = np.array([1., 1., 1., 1.])

nuevos_colores = np.zeros(shape=(n_clases, 4))
div = 256/6.
nuevos_colores[0]=sombra
nuevos_colores[1]=sombra_sa
nuevos_colores[2]=agua
nuevos_colores[3]=nieve
nuevos_colores[4]=tierra
nuevos_colores[5]=nube
custom_cm = ListedColormap(nuevos_colores) 

#%%
for i, img in enumerate(imgs_train):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img)
    imsh = axs[1].imshow(mask_train[i], cmap=custom_cm, vmin=0, vmax=6)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title('Imagen de entrenamiento %d' % i)
    axs[1].set_title('Máscara')
    patches = [ mpatches.Patch(color=nuevos_colores[i], label=clases[i]) for i in range(len(clases)) ]
    legend = plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), 
                        loc=2, borderaxespad=0., 
                        facecolor = [0.1,0.1,0.1,1.])
    plt.show()
##%%
#for img in imgs_test:
#    fig = plt.figure()
#    plt.imshow(img)
#    plt.axis('off')
#    plt.title('Imágen de prueba %i' % i)
#    plt.show()
#%%    
for i, img in enumerate(imgs_test):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img)
    imsh = axs[1].imshow(mask_test[i], cmap=custom_cm, vmin=0, vmax=6)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title('Imagen de prueba %d' % i)
    axs[1].set_title('Resultado')
    patches = [ mpatches.Patch(color=nuevos_colores[i], label=clases[i]) for i in range(len(clases)) ]
    legend = plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), 
                        loc=2, borderaxespad=0., 
                        facecolor = [0.1,0.1,0.1,1.])
    plt.show()