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
    # covarianzas = np.zeros(np.matrix(
    #                 np.zeros(
    #                     shape=(datos_shape[3], datos_shape[3])))
    #                     for c in clases])
    for i_img in range(datos_shape[0]):
        for x in range(datos_shape[1]):
            for y in range(datos_shape[2]):
                k = datos_y[i_img, x, y]
                m_temp = np.matrix(datos_x[i_img, x, y]-medias[k])
                covarianzas[k]+=np.matmul(m_temp.T, m_temp)
    for k, clase in enumerate(clases):
        covarianzas[k]/=(prob_clases[k]*total_datos)
    return covarianzas

def disc_bayes(x, m, SI, detS, Pk):
    temp = np.matrix(x-m)
    SI = np.matrix(SI)
    temp2 = np.matmul(temp, SI)
    temp3 = np.matmul(temp2, temp.T).item()
    #print(temp3)
    disc = -(1.0/2.0)*temp3-(1.0/2.0)*np.log(detS)+np.log(Pk)
    #print(disc)
    return disc.item()

def predecir(datos_x, modelo):
    datos_shape = datos_x.shape
    #datos_x.reshape(datos_shape[0]*datos_shape[1], datos_shape[2])
    prediccion_y = np.empty_like(datos_x[:,:,0]).astype(np.uint8)
    discr = np.empty(len(modelo['clases'])).astype(np.float64)
    for i in range(datos_shape[0]):
        for j in range(datos_shape[1]):
            for k in modelo['clases']:
                x=datos_x[i,j]
                m=modelo['medias'][k]
                S=modelo['covarianzas'][k]
                SI=modelo['covs_inv'][k]
                detS=modelo['det_covs'][k]
                pk=modelo['prob_a_priori'][k]
                discr[k] = disc_bayes(x, m, SI, detS, pk)
            prediccion_y[i,j] = discr.argmax()
            #print(prediccion_y[i,j])
    return prediccion_y