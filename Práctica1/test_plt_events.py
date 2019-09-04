from matplotlib import pyplot as plt
import cv2
from threading import Semaphore
import numpy as np

def find_region(x, y):
    #print '\nloop runs till region growing is complete'
    #print 'starting points',i,j
    region_points = []
    region_points.append([x,y])

    img_rg = np.zeros([img_bin.shape[0]+1, img_bin.shape[1]+1])
    img_rg[x, y] = 255.0

    count = 0
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]

    while( len(region_points)>0):
        if count == 0:
            point = region_points.pop(0)
            i = point[0]
            j = point[1]
        #print '\nloop runs till length become zero:'
        #print 'len',len(region_points)
        #print 'count',count 
        val = img_bin[i][j]
        lt = val - 8
        ht = val + 8
        #print 'value of pixel',val
        for k in range(8):	
            #print '\ncomparison val:',val, 'ht',ht,'lt',lt
            if img_rg[i+x[k]][j+y[k]] !=1:
                try:
                    if  img_bin[i+x[k]][j+y[k]] > lt and img_bin[i+x[k]][j+y[k]] < ht:
                        #print '\nbelongs to region',img_bin[i+x[k]][j+y[k]]
                        img_rg[i+x[k]][j+y[k]]=1
                        p = [0,0]
                        p[0] = i+x[k]
                        p[1] = j+y[k]
                        if p not in region_points: 
                            if 0< p[0] < img_bin.shape[0] and 0< p[1] < img_bin.shape[0]:
                                ''' adding points to the region '''
                                region_points.append([i+x[k],j+y[k]])
                    else:
                        #print 'not part of region'
                        img_rg[i+x[k]][j+y[k]]=0
                except IndexError:     
                    continue

    #print '\npoints list',region_points
    point = region_points.pop(0)
    i = point[0]
    j = point[1]
    count = count +1
    return img_rg
    #find_region(point[0], point[1])		


def obtener_vecinos(x, y):
    vec_x = [-1, 0, 1, -1, 1, -1, 0, 1]
    vec_y = [-1, -1, -1, 0, 0, 1, 1, 1]
    for k in range(8):
        if(img_bin[x,y] == img_bin[x+vec_x[k],y+vec_x[y]]):
            pass
    vecinos = []
    return vecinos

def region_growing():
    region = np.zeros_like(img_bin)
    vecinos = obtener_vecinos()
    for xy_vecino in vecinos:
        pass
    return region

def onmouseclick(event):
    if(event.inaxes == ax1):
        with m_seed:
            seed = [np.round(event.xdata).astype(np.int), np.round(event.ydata).astype(np.int)]
            print(seed)
        img_rg = find_region(seed[0], seed[1])
        #ax2.imshow(np.random.randint(low=0, high=255, size=img_bin.shape, dtype=np.uint8))
        ax2.imshow(img_rg, cmap='gray')
        fig1.canvas.draw()

seed = []
m_seed = Semaphore(1)

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