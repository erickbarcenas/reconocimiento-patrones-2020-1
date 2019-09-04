# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:06:48 2019

@author: Andres
"""

#%%
import cv2
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

#%%
midbrain = spio.loadmat('.\\Midbrain.mat')

#%%
img_midbrain = np.array(midbrain['midbrain'])
img_midbrain_thresh = np.array(midbrain['midbrainthesh'])
brain_shape = img_midbrain_thresh.shape

#%%
plt.title('Midbrain original')
plt.imshow(img_midbrain, cmap='gray')
plt.show()

#%%
plt.title('Midbrain threshold + noise')
plt.imshow(img_midbrain_thresh, cmap='gray')
plt.show()

#%%
# Generando la operación inversa
# x = (8/3)y si 0<y<0.25
# x = y si 0.4<y<0.6
# x = (8/3)y-1 si 0.6<y<0.75
img_decontrastada = np.zeros(shape = brain_shape)

#%%
for ii in range(brain_shape[0]):
    for jj in range(brain_shape[1]):
        if(img_midbrain_thresh[ii,jj] < 0.25):
            img_decontrastada[ii,jj] = (8/5)*img_midbrain_thresh[ii,jj]
        elif (img_midbrain_thresh[ii,jj] >= 0.25 and img_midbrain_thresh[ii,jj] <0.6):
            img_decontrastada[ii,jj] = img_midbrain_thresh[ii,jj]
        elif img_midbrain_thresh[ii,jj] >= 0.6 and img_midbrain_thresh[ii,jj]<=0.75:
            img_decontrastada[ii,jj] = (8/3)*img_midbrain_thresh[ii,jj]-1
# Quitar valores negativos
img_decontrastada = np.add(img_decontrastada, -np.amin(img_decontrastada))

#%%
plt.title('Midbrain decontrastada con ruido')
plt.imshow(img_decontrastada, cmap='gray', vmin='0', vmax='1')
plt.show()

#%%
# Pasar todo a enteros de 8 bits
img_decontrastada = np.rint(np.multiply(img_decontrastada, 255)).astype(np.uint8)

#%%
# Filtrar imágenes con diferentes filtros
img_filtrada3 = cv2.medianBlur(img_decontrastada, 3)
img_filtrada5 = cv2.medianBlur(img_decontrastada, 5)
img_filtrada9 = cv2.medianBlur(img_decontrastada, 9)
img_filtrada13 = cv2.medianBlur(img_decontrastada, 13)

#%%
fig = plt.figure(figsize = (10,10))
fig.suptitle('Imagenes con filtro de mediana a diferentes tamaños', fontsize=14)

ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Filtro 3x3')
ax1.imshow(img_filtrada3, cmap='gray', vmin='0', vmax='255')

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Filtro 5x5')
ax2.imshow(img_filtrada5, cmap='gray', vmin='0', vmax='255')

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Filtro 9x9')
ax3.imshow(img_filtrada9, cmap='gray', vmin='0', vmax='255')

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('Filtro 13x13')
ax4.imshow(img_filtrada13, cmap='gray', vmin='0', vmax='255')

plt.subplots_adjust(hspace=0.5)
plt.show()

#%%
plt.title('Imagen a segmentar')
plt.imshow(img_filtrada3, cmap='gray', vmin='0', vmax='255')
plt.show()

#%%
# Equalización local del histograma, con ventanas de 8x8
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
plt.title('Imagen equalizada')
img_eq = clahe.apply(img_filtrada3)
plt.imshow(img_eq, cmap='gray')
plt.show()

#%%
# Aplicar umbralización para quitar zonas que no necesito
ret,img_bin = cv2.threshold(img_filtrada3,100,255,cv2.THRESH_BINARY)
plt.title('Imagen umbralizada')
plt.imshow(img_bin, cmap='gray')
plt.show()

#%%
# Obtener la plantilla
template = cv2.imread('./template2.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
plt.title('Template')
plt.imshow(template, cmap='gray')
plt.show()

#%%
result = cv2.matchTemplate(image = img_bin, templ = template, method=cv2.TM_CCOEFF_NORMED)
result = np.abs(result)**3
val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
plt.imshow(result8, cmap='gray')

#%%
""" 
# Suavizar bordes
img_suav = cv2.GaussianBlur(img_eq,(3,3),0)
plt.title('Imagen suavizada')
plt.imshow(img_suav, cmap='gray')

"""