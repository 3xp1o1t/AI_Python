#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:43:55 2019

@author: JMontiel
"""
# In[6]:
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as FileDialog
# In[7]:
# Importar archivo con datos
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

# In[8]
# Configuracion del algoritmo
atributos = conjunto[:, :-1]
clases = conjunto[:, -1]

K = 10
[ren, col] = conjunto.shape
tolerancia = 0.00001
iteraciones = 1200
j = 0
umbrales = np.zeros((1, iteraciones))
valores_j = np.zeros((1, iteraciones))

# In[9]
# Colores
colores = np.random.random_sample((K,3))
colores_t = tuple(map(tuple,colores))

# In[10]
# Normalizar el conjunto
[ra, ca] = atributos.shape # ra > renglones atributos - ca > olumnas atributos
maximos = atributos.max(0)
conjunto_normalizado = np.zeros([ra, ca])
maximos_array = np.tile(maximos, (ra, 1))
conjunto_normalizado[:, :ca] = atributos[:, :ca] / maximos_array
# Redondear a 4 decimales n.0000
conjunto_normalizado = np.around(conjunto_normalizado, decimals=4)
# In[11]
# Tranponer el conjunto
conjunto_t = conjunto_normalizado.T
# Generar numeros aleatorios en un rango del total de filas (4769)
aleatorios = np.random.permutation(range(ren))
# Reshape aleatorios para obtener K centroides al azar del conjunto transpuesto
aleatorios = np.reshape(aleatorios, (1, ren))
# Tomar K centroides al azar
centroides = conjunto_t[:, aleatorios[0, :K]]

# In[12]
# Configuracion para la distancia euclidiana y los nuevos centroides
nuevos_centroides = np.zeros((col-1, K))
distancia_euclidiana = np.zeros((K, ren))

# In[13]
def distanciaEuclidiana(conjunto_transpuesto, centroides_remapeados):
    distancia_euclidiana = np.sqrt(sum((conjunto_transpuesto - centroides_remapeados)**2, 1))
    return np.around(distancia_euclidiana, decimals=4)

# In[14]
contador = 1
# Inicio del algoritmo
while(contador <= iteraciones):
    for i in range(0, K):
        # Replicar la matriz de centroides
        centroides_r = np.array([centroides[:, i],] * ren).transpose()
        # Calcular la distancia euclidiana - norm
        distancia_euclidiana[i, :] = distanciaEuclidiana(conjunto_t, centroides_r)
    
    # min(0) obtiene el minimo de cada columna.
    minimos_r = np.tile(distancia_euclidiana.min(0), (K, 1))
    
    # u es la matriz de pertencias
    u = (minimos_r == distancia_euclidiana).astype(int)
    
    # Calcular los nuevos centroides
    nuevos_centroides = np.zeros((col-1, K))
    for i in range(0, K):
        centroides_r = np.tile(u[i, :], (2, 1))
        nuevos_centroides[:, i] = np.sum(centroides_r * conjunto_t, 1) / np.sum(u[i, :])
    
    
    print('Salir?: ', (centroides == nuevos_centroides).all())
    
    costo = sum(sum(distancia_euclidiana))
    
    # Existen 2 posibles salidas
    # 1-. si los centroides no cambian, se puede salir
    # 2-. si el costo no cambia, se puede salir.
    if (centroides == nuevos_centroides).all():
        break
    
    #if costo == j:
    #    break;
    
    umbrales[0, contador-1] = costo - j
    
    j = costo
    
    valores_j[0, contador-1] = j
    
    centroides = nuevos_centroides
    
    print('Iteacion: ', contador, ' Costo: ', costo)
    contador = contador + 1
    
# In[15]:
# Centroides colocados en el conjunto
plt.subplot(2, 2, 1)
plt.title('Centroides en el conjunto')
plt.xlabel('Atributos 1')
plt.ylabel('Atributos 2')
plt.scatter(conjunto_t[0, :], conjunto_t[1, :], 10)
for i in range(K):
    plt.scatter(centroides[0, i], centroides[1, i], 70, marker='o', color=colores_t[i])

# In[16]:
# Conjunto segmentando acorde el centroide
plt.subplot(2, 2, 2)
plt.title("Conjunto segmentado")
plt.xlabel("Atributos 1")
plt.ylabel("Atributos 2")
for i in range(K):
  cols, = np.where(u[i, :])
  plt.scatter(conjunto_t[0, cols], conjunto_t[1, cols], color=colores_t[i])
  
# In[17]:
# Historial del costo de la funcion J
plt.subplot(2, 2, 3)
plt.title('Costo de la funcion')
plt.xlabel('Iteraciones')
plt.ylabel('Funcion J')
plt.plot(range(0,np.size(umbrales[0, :contador-1])), umbrales[0, :contador-1], 'rx',
         range(0,np.size(valores_j[0, :contador-1])), valores_j[0, :contador-1], 'b*',linestyle='dashed', 
         linewidth=2, markersize=5)

# In[18]
# Atributos en total por agrupacion o cluster.
plt.subplot(2, 2, 4)
plt.title("Atributos por cluster")
plt.xlabel("Cluster")
plt.ylabel("Atributos")
for i in range(K):
  cols, = np.where(u[i, :])
  plt.bar(i, np.size(conjunto_t[0, cols]), color=colores_t[i])
  
plt.show()