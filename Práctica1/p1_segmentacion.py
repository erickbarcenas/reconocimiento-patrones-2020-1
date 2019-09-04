#%%
import cv2
import numpy as np
#import scipy.io as spio
#import matplotlib.pyplot as plt
#%%
# Usar region growing para obtener la región deseada
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
    list = []
    outimg = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while(len(list) > 0):
        pix = list[0]
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
    return outimg

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), img_bin[y,x])
        clicks.append((y,x))

#%%
img_bin = cv2.imread('./imagen_preprocesada.bmp', 0)
clicks = []
cv2.namedWindow('Input',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Input', on_mouse, 0, )
cv2.imshow('Input', img_bin)
cv2.waitKey()
seed = clicks[-1]
out = region_growing(img_bin, seed)
cv2.imshow('Region Growing', out)
cv2.waitKey()
cv2.destroyAllWindows()

# #%%
# # Obtener la plantilla
# template = cv2.imread('./template2.png')
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# plt.title('Template')
# plt.imshow(template, cmap='gray')
# plt.show()

# #%%
# result = cv2.matchTemplate(image = img_bin, templ = template, method=cv2.TM_CCOEFF_NORMED)
# result = np.abs(result)**3
# val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
# result8 = cv2.normalize(result,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
# plt.imshow(result8, cmap='gray')

#%%
""" 
# Suavizar bordes
img_suav = cv2.GaussianBlur(img_eq,(3,3),0)
plt.title('Imagen suavizada')
plt.imshow(img_suav, cmap='gray')

"""