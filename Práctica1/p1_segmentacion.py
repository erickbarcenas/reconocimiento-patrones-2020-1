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
    minx = 100000
    miny = 100000
    maxx = 0
    maxy = 0
    total_px = 0
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
                
                #Encontrar las dimensiones y los pixeles que ocupa
                minx = coord[0] if coord[0]<minx else minx
                miny = coord[1] if coord[1]<miny else miny
                maxx = coord[0] if coord[0]>maxx else maxx
                maxy = coord[1] if coord[1]>maxy else maxy
                total_px += 1

                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
    print('Termina el proceso...')
    dimx = maxx-minx
    dimy = maxy-miny
    return outimg, [dimx, dimy], total_px

#%%
def onmouseclick(event):
    if(event.inaxes == ax1):
        #s.acquire()
        seed = [np.round(event.ydata).astype(np.int), np.round(event.xdata).astype(np.int)]
        print(seed)
        img_rg, dim_img, area_img = region_growing(img_bin, seed)
        #ax2.imshow(np.random.randint(low=0, high=255, size=img_bin.shape, dtype=np.uint8))
        ax2.imshow(img_rg, cmap='gray')
        textstr = '\n'.join((
            'Ancho: %.1f mm' % (dim_img[1]*0.5),
            'Alto: %.1f mm' % (dim_img[0]*0.5),
            'Área: %.2f mm\u00b2' % (area_img*0.5**2)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        #text_box.set_text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
        #            verticalalignment='top', bbox=props)
        text_box.set_text(textstr)
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
ax2.imshow(np.zeros([img_bin.shape[0], img_bin.shape[1]]).astype(np.uint8), cmap='gray')

textstr = '\n'.join((
    'Ancho: 0 mm',
    'Alto: 0 mm',
    'Área: 0 mm\u00b2'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
text_box = ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.show()