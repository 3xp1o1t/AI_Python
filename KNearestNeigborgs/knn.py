#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 02:16:15 2019

@author: JMontiel
"""
# In[3]
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as FileDialog

# In[4]
# Configuracion del algoritmo
ruta = FileDialog.askopenfilename(
    initialdir="E:/IA/", 
    filetypes=(
        ("Archivo de datos", "*.dat"),
        ("Archivo de texto", "*.txt"),
        ("Todos los ficheros","*.*")
    ), 
    title = "Abrir un fichero."
)
archivo = ruta
conjunto = np.genfromtxt(archivo, delimiter=',')
atributos = conjunto[:, :-1]
clases = conjunto[:, -1]
[ren, col] = conjunto.shape
K = 10

# In[5]
# Funcion de normalizacion para los datos
def normalizar(atributos):
    [ren, col] = atributos.shape
    maximos = atributos.max(0)
    conjunto_normalizado = np.zeros([ren, col])
    maximos_array = np.tile(maximos, (ren, 1))
    conjunto_normalizado[:, :col] = atributos[:, :col] / maximos_array
    return np.around(conjunto_normalizado, decimals=4)

# In[6]
# Funcion para calcular la distancia euclidiana
def distanciaEuclidiana(atributos, array_observacion):
    distEUC = np.sqrt(np.sum((atributos - array_observacion)**2, 1))
    distEUC = distEUC.reshape((ren, 1))
    return np.around(distEUC, decimals=4)

# In[7]
# Normalizacion del conjunto y ploteo de datos
conjunto_normalizado = normalizar(atributos)

plt.scatter(conjunto_normalizado[:, 0], conjunto_normalizado[:, 1])
plt.show()
# In[8]
# column_stack para aderir las clases al conjunto
conjunto_normalizado = np.column_stack((conjunto_normalizado, clases))
# Mostrar los valores pertenecientes
for i in range(-1, 2):
    posicion, = np.where(conjunto_normalizado[:, 2] == i)
    pertenecientes = conjunto_normalizado[posicion, :]
    plt.scatter(pertenecientes[:, 0], pertenecientes[:, 1], 10)
    
# In[9]
# Obtener la nueva observacion para calcular sus vecinos mas cercanos
nueva_obs = plt.ginput(1)
observacion_array = np.tile(nueva_obs[0], (ren, 1))
plt.scatter(observacion_array[0, 0], observacion_array[0, 1], 30)

distEuc = distanciaEuclidiana(conjunto_normalizado[:, :col-1], observacion_array)
posicion = np.argsort(distEuc[:, 0])

# In[10]
# Plotear la nueva observacion con sus vecinos mas cercanos
for i in range(0, K):
    plt.plot([observacion_array[i, 0], conjunto_normalizado[posicion[i], 0]], [observacion_array[i, 1], conjunto_normalizado[posicion[i], 1]])     

plt.show()
