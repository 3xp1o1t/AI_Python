#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 18:12:19 2019

@author: JMontiel
"""
# In[6]:

import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog as FileDialog

# In[7]:

# Configuracion del algoritmo
K = 3
tolerancia = 0.00001
iteraciones = 1200
j = 0
contador = 1


# In[8]:
# Ruta imagen
ruta = FileDialog.askopenfilename(
    initialdir="E:/IA/", 
    filetypes=(
        ("Imagen jpg", "*.jpg"),
        ("Imagen png", "*.png"),
        ("Todos los ficheros","*.*")
    ), 
    title = "Abrir un fichero."
)

# In[9]:

img = plt.imread(ruta)

# In[10]:

# Separar los colores de la imagen
# r(renglones) c(columnas) ch(channels-canales)
r, c, ch = img.shape
# ren x cols
rc = r * c
# colores rgb
R = np.reshape(img[:, :, 0], (1, rc))
G = np.reshape(img[:, :, 1], (1, rc))
B = np.reshape(img[:, :, 2], (1, rc))
# Reshape para restructurar la matriz, si no se aplica devuelve (ch, 1, rc)
rgb = np.array([R, G, B]).reshape((3, rc))

# In[11]:

# Prealocar memoria
centroides = np.zeros((ch, K))
distancia_euclidiana = np.zeros((K, rc))
img_tratada = np.zeros((ch, rc))

# In[12]:

# Matriz de particiones difusa
u = np.around(np.random.rand(K, rc), decimals=4)
# En lugar de reshape, usar tile para aplicar la operacion sobre cada columna.
divisor = np.around(np.tile(sum(u), (K, 1)), decimals=4)
# Obtener n numeros que sumados den 1.0 por columna
u = u / divisor


# In[13]:
def distanciaEuclidiana(_rgb_, centroides_remapeados):
    distancia_euclidiana = np.sqrt(sum((_rgb_ - centroides_remapeados)**2, 1))
    return np.around(distancia_euclidiana, decimals=4)
# In[14]:

# Ejecucion del algoritmo
while(contador <= iteraciones):
    u_cuadrada = np.around(u ** 2, decimals=4)
    for i in range(K):
        cluster = np.tile(u_cuadrada[i, :], (ch, 1))
        cen_x_xn = cluster * rgb
        centroides[:, i] = np.sum(cen_x_xn, 1) / np.sum(u_cuadrada[i, :])
        
        # Usar tile no funciona, remapea mal y reshape no puede cambiar la estructura.
        centroide_remapeado = np.array([centroides[:, i],] * rc).transpose()
        distancia_euclidiana[i, :] = distanciaEuclidiana(rgb, centroide_remapeado)
    
    for i in range(K):
        distancia_rc = np.tile(distancia_euclidiana[i, :], (K, 1))
        u[i, :] = np.around(np.ones(1) / (sum((distancia_rc / distancia_euclidiana)**2)), decimals=4)
    
    
    costo = sum(sum((u_cuadrada * distancia_euclidiana)**2))
    diferencia = abs(j - costo)
    j = costo
    
    if diferencia < tolerancia:
        print('Iteracion: ', contador, ' de ', iteraciones, ' Umbral: ', diferencia, ' Fun Obj: ', costo)
        break;
    
    print('Iteracion: ', contador, ' de ', iteraciones, ' Umbral: ', diferencia, ' Fun Obj: ', costo)
    contador = contador + 1


# In[15]:
# Imagen original
plt.subplot(2, 1, 1)
plt.title('Imagen Normal')
plt.imshow(img)

# In[16]:
# imagen clusterizada
plt.subplot(2, 2, 3)
pos = np.argmax(u, axis=0).reshape(1, rc)
img_tratada = np.fix(centroides[:, pos])
img_tratada_cpy = img_tratada.copy()
# Reconstruir la imagen convirtiendo sus valores a enteros.
img_tratada = np.reshape(img_tratada.transpose(), (r, c, ch)).astype(int)
plt.title('Imagen Segmentada')
plt.imshow(img_tratada)

# In[17]:
# Puntos simulados con GInput abarcan linea blanca
punto = plt.ginput(1)
p_elegido = img_tratada[round(int(punto[0][1])), round(int(punto[0][0])), :]
p_elegido = np.reshape(p_elegido, (3, 1))
unos = np.tile(p_elegido, (1, rc))
#iguales = (img_tratada_cpy == unos).astype(int)
img_tratada_cpy = img_tratada_cpy.astype(int)
img_tratada_cpy = np.reshape(img_tratada_cpy, (3, rc))
iguales = np.isclose(img_tratada_cpy, unos).astype(int)
sumados = sum(iguales).reshape(1, rc)
valor_2 =  np.where(sumados == 3)
cyan = np.array([[0],[255],[255]])
fondo = np.tile(cyan, (1, len(valor_2[1]))).astype(int)

# In[18]:
# Remplazar los pixeles por el color cyan
img_tratada_cpy[:, valor_2[1]] = fondo

# In[19]:
# Reconstruir la imagen
img_tratada_cpy = np.reshape(img_tratada_cpy.transpose(), (r, c, ch))

# In[20]:
plt.subplot(2, 2, 4)
plt.title('Pixeles a cyan')
plt.imshow(img_tratada_cpy)

plt.show()
