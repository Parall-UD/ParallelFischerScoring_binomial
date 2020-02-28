# -*- coding: utf-8 -*-
print 'Importando librerias...'
import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import sys, getopt
import time
from numpy.linalg import norm
print 'Librerias importadas correctamente'

# Parámetros usuario
input_dataset = sys.argv[1]

#Lectura de datos
data = pd.read_csv(input_dataset)
y = data.iloc[:,0:1].values
x = data.iloc[:,1:].values
x = np.ascontiguousarray(x)
y = np.ascontiguousarray(y)
y = np.reshape(y,(y.shape[0]))

#Función iterativa sobre la matrix X
def sapply(x_matrix,beta):
    x_matrix_aux = np.zeros((x_matrix.shape[0]))
    x_matrix_aux_x = x_matrix.copy()
    for i in range(x_matrix.shape[0]):
        x_matrix_aux_x[i] = 1/(1+np.exp(-(np.dot(x_matrix_aux_x[i,],beta))))
        x_matrix_aux[i] = x_matrix_aux_x[i][0]
    return x_matrix_aux.T

# Función para reemplazar valores nulos
def nanValue(array):
    array[np.isnan(array)]=1.000000
    return array

# Función logística
def logis(y,x):
    end = 0
    start = 0
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    start=time.time()
    # Translado de variable a GPU
    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    linalg.init()
    # Transpuesta de X
    x_gpu_T = linalg.transpose(x_gpu)
    beta_gpu = linalg.dot(linalg.dot(linalg.inv(linalg.dot(x_gpu_T,x_gpu)),x_gpu_T),y_gpu)
    j = 1
    while(True):
        mu = sapply(x,beta_gpu.get())
        mu = mu.astype(np.float32)
        mu_gpu = gpuarray.to_gpu(mu)
        V_gpu= linalg.diag(mu_gpu)
        f2_gpu = linalg.multiply(mu_gpu,1-mu_gpu)
        f3_gpu = linalg.diag(1/f2_gpu)
        f4_gpu = (y_gpu-mu_gpu)
        f5_gpu = linalg.dot(f3_gpu,f4_gpu)
        if(np.isnan(f5_gpu.get()).any()):
            f5_cpu = f5_gpu.get()
            f5_cpu = nanValue(f5_cpu)
            f5_gpu = gpuarray.to_gpu(f5_cpu.astype(np.float32))
        y_1_gpu = linalg.dot(x_gpu,beta_gpu) + f5_gpu
        beta_1_gpu = linalg.dot(linalg.dot(linalg.dot(linalg.inv(linalg.dot(linalg.dot(x_gpu_T,V_gpu),x_gpu)),x_gpu_T),V_gpu),y_1_gpu)
        check_value = np.absolute(linalg.norm(beta_1_gpu-beta_gpu))
        #if(check_value<0.00001):
            #break
        if(j == 10 or check_value<0.00001):
            break
        beta_gpu = beta_1_gpu
        j = j + 1
    end = time.time()
    tiempo = (end-start)
    return {"iteraciones":j,"Betas":beta_gpu.get(),"time":tiempo}

result = logis(y,x)

print result
