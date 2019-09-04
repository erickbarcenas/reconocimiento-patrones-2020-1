#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Semaphore
#import scipy.io as spio
#import matplotlib.pyplot as plt
#%%
# Usar region growing para obtener la región deseada
def obtener_8_vecinos(x, y, shape):
    #print('Onteniendo vecinos...')
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #print('Vecinos obtenidos: ', out)
    return out

def region_growing(img, seed):
    print('Inicia el proceso...')
    list = []
    outimg = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while(len(list) > 0):
        pix = list[0]
        #print('Analizando pixel: ', pix)
        outimg[pix[0], pix[1]] = 255
        for coord in obtener_8_vecinos(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
        #cv2.imshow("progress",outimg)
        #cv2.waitKey(1)
    print('Termina el proceso...')
    return outimg

#%%
def onmouseclick(event):
    if(event.inaxes == ax1):
        #s.acquire()
        seed = [np.round(event.ydata).astype(np.int), np.round(event.xdata).astype(np.int)]
        print(seed)
        img_rg = region_growing(img_bin, seed)
        #ax2.imshow(np.random.randint(low=0, high=255, size=img_bin.shape, dtype=np.uint8))
        ax2.imshow(img_rg, cmap='gray')
        fig1.canvas.draw()
        #s.release()

#%%
#s = Semaphore(1)
img_bin = cv2.imread('./imagen_preprocesada.bmp', 0)
fig1 = plt.figure(figsize=(10, 7))
fig1.canvas.mpl_connect('button_press_event', onmouseclick)

ax1 = fig1.add_subplot(121)
ax1.set_title('Selecciona un área para medirla')
ax1.imshow(img_bin, cmap='gray')
ax1.axis('off')

ax2 = fig1.add_subplot(122)
ax2.set_title('Resultado')
ax2.axis('off')
ax2.imshow(img_bin, cmap='gray')

textstr = '\n'.join((
    'Ancho: %d mm',
    'Alto: %d mm',
    'Área: %d mm\u00b2'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.show()