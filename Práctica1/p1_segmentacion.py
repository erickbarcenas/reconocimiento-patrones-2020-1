#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
# Usar region growing para obtener la región deseada
# Implementación actual: https://stackoverflow.com/questions/43923648/region-growing-python
# Implementación a futuro: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/medium/segment/recurse.htm
def obtener_8_vecinos(x, y, shape):
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
    
    return out

def region_growing(img, seed):
    print('Inicia el proceso. Espere por favor...')
    minx = 100000
    miny = 100000
    maxx = 0
    maxy = 0
    total_px = 0
    lista = []
    outimg = np.zeros_like(img)
    lista.append((seed[0], seed[1]))
    processed = []
    while(len(lista) > 0):
        pix = lista[0]
        outimg[pix[0], pix[1]] = 255
        for coord in obtener_8_vecinos(pix[0], pix[1], img.shape):
            if img[coord[0], coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                
                #Encontrar las dimensiones y los pixeles que ocupa
                minx = coord[0] if coord[0]<minx else minx
                miny = coord[1] if coord[1]<miny else miny
                maxx = coord[0] if coord[0]>maxx else maxx
                maxy = coord[1] if coord[1]>maxy else maxy
                
                if not coord in processed:
                    total_px += 1
                    lista.append(coord)
                processed.append(coord)
        lista.pop(0)
    print('Termina el proceso...')
    dimx = maxx-minx
    dimy = maxy-miny
    return outimg, [dimx, dimy], total_px

#%%
def onmouseclick(event):
    if(event.inaxes == ax1):
        seed = [np.round(event.ydata).astype(np.int), np.round(event.xdata).astype(np.int)]
        img_rg, dim_img, area_img = region_growing(img_bin, seed)
        ax2.imshow(img_rg, cmap='gray')
        textstr = '\n'.join((
            'Ancho: %.1f mm' % (dim_img[1]*0.5),
            'Alto: %.1f mm' % (dim_img[0]*0.5),
            'Área: %.2f mm\u00b2' % (area_img*0.25)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text_box.set_text(textstr)
        fig1.canvas.draw()

#%%
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