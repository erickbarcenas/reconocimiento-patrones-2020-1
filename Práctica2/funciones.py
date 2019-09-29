import numpy as np
#from tqdm import tqdm

def obtenerProbAPriori(datos_y, n_clases : int):
    data_shape = datos_y.shape
    total_datos = np.multiply.reduce(data_shape)
    hist = np.histogram(datos_y.reshape(1, total_datos), 
                    bins=n_clases, 
                    range=[0,n_clases-1])
    prob = hist[0].astype(np.long) / total_datos
    return prob

def obtenerMatricesMedias(datos_x, datos_y, clases, prob_clases):
    datos_shape = datos_x.shape
    total_datos = np.multiply.reduce(datos_shape)
    medias = np.zeros(shape=(len(clases), datos_x.shape[3]))
    for i_img in range(datos_shape[0]):
        #tqdm.write('Calculando matriz de medias de la imagen %d' % i_img)
        for x in range(datos_shape[1]):
            for y in range(datos_shape[2]):
                k = datos_y[i_img, x, y]
                medias[k]+=datos_x[i_img, x, y]
    for k, clase in enumerate(clases):
        medias[k]/=(prob_clases[k]*total_datos)
    return medias

def obtenerMatricesCovarianzas(datos_x, datos_y, medias,clases, prob_clases):
    datos_shape = datos_x.shape
    total_datos = np.multiply.reduce(datos_shape)
    covarianzas = np.zeros(shape=(len(clases), datos_shape[3], datos_shape[3]))
    for i_img in range(datos_shape[0]):
        for x in range(datos_shape[1]):
            for y in range(datos_shape[2]):
                k = datos_y[i_img, x, y]
                m_temp = datos_x[i_img, x, y]-medias[k]
                covarianzas[k]+=(m_temp*np.transpose(m_temp))
    for k, clase in enumerate(clases):
        covarianzas[k]/=(prob_clases[k]*total_datos)
    return covarianzas