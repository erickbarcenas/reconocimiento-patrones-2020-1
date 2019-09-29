# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:03 2019

@author: Andres
"""
#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob

#%%
path_datos_prueba = glob.glob('./Dataset/Test/*data.tif')
datos_prueba  = np.array([np.dstack(np.array(rasterio.open(path).read())) for path in path_datos_prueba])
#%%
imgs = np.flip(datos_prueba[:,:,:,1:4]/65535.0, 3)
#%%
modelo = np.load('./modelo.npy', allow_pickle=True).item()

#%%
for i, img in enumerate(imgs):
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(imgs[i])
    axs[1].imshow(np.random.randn(1000,1000))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title('Imagen %d' %i)
    axs[1].set_title('Resultados')
    plt.show()