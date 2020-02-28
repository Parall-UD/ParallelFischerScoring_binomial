# -*- coding: utf-8 -*-
print 'Importando librerias...'
import pandas as pd
import numpy as np
from numpy.linalg import inv, norm
import time
import sys, getopt
print 'Librerias importadas correctamente'

# Parámetros de usuario
input_dataset = sys.argv[1]

#Lectura de datos
data = pd.read_csv(input_dataset)
y = data.iloc[:,0:1].values
x = data.iloc[:,1:].values
x = np.ascontiguousarray(x)
y = np.reshape(y,(y.shape[0]))

#Función iterativa sobre la matrix X
def sapply(x_matrix,beta):
    x_matrix_aux = np.zeros((x_matrix.shape[0]))
    x_matrix_aux_x = x_matrix.copy()
    for i in range(x_matrix.shape[0]):
        x_matrix_aux_x[i] = 1/(1+np.exp(-(np.dot(x_matrix_aux_x[i,],beta))))
        x_matrix_aux[i] = x_matrix_aux_x[i][0]
    return x_matrix_aux.T

# Función logística
def logis(y,x):
    end = 0
    start = 0
    start=time.time()
    beta = np.dot(np.dot(inv(np.dot(x.T,x)),x.T),y)
    j = 1
    while(True):
        mu = sapply(x,beta)
        V = np.diag(mu)
        #print mu
        a = np.multiply(mu,1-mu)
        #for i in range(a.shape[0]):
            #if(a[i] == 0):
                #print 'cero'
                #print i
        #print a
        f1 = np.diag(1/a)
        y_1 = np.dot(x,beta) + (np.dot(f1,(y-mu)))
        beta_1 = np.dot(np.dot(np.dot(inv(np.dot(np.dot(x.T,V),x)),x.T),V),y_1)
        check_value = np.absolute(norm(beta_1-beta))
        #if(check_value<0.00001):
            #break
        if(j == 10 or check_value<0.00001):
            break


        beta = beta_1
        j = j + 1
    end = time.time()
    tiempo = (end-start)
    return {"iteraciones":j,"Betas":beta, "time":tiempo}

result = logis(y,x)
print result
