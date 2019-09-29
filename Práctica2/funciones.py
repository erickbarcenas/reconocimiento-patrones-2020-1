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
                m_temp = np.matrix(datos_x[i_img, x, y]-medias[k])
                covarianzas[k]+=np.matmul(m_temp.T, m_temp)
    for k, clase in enumerate(clases):
        covarianzas[k]/=(prob_clases[k]*total_datos)
    return covarianzas

def disc_bayes(x, m, S, Pk):
    temp = np.matrix(x-m)
    S = np.matrix(S)
    temp2 = np.matmul(temp, S)
    temp3 = np.matmul(temp2, temp.T).item()
    disc = -(1.0/2.0)*temp3-(1.0/2.0)*np.log(np.linalg.det(S))+np.log(Pk)
    #print(disc)
    return disc.item()

def predecir(datos_x, modelo):
    datos_shape = datos_x.shape
    #datos_x.reshape(datos_shape[0]*datos_shape[1], datos_shape[2])
    prediccion_y = np.empty(datos_shape).astype(np.uint8)
    discr = np.empty(len(modelo['clases'])).astype(np.float64)
    for i in range(datos_shape[0]):
        for j in range(datos_shape[1]):
            for k in modelo['clases']:
                x=datos_x[i,j]
                m=modelo['medias'][k]
                S=modelo['covarianzas'][k]
                pk=modelo['prob_a_priori'][k]
                discr[k] = disc_bayes(x, m, S, pk)
            prediccion_y[i,j] = discr.argmax()
    return prediccion_y