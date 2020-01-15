#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 4 14:56:00 2020

@author: JMontiel
"""
# In[88]:


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from tkinter import filedialog as FileDialog


# In[89]:


# Cargar el dataset
archivo = FileDialog.askopenfilename(
    initialdir="C:",
    filetypes=(
        ("Archivos de Texto", "*.txt"),
        ("Archivos de datos", "*.dat"),
        ("Todos los ficheros", "*.*")
    ),
    title = 'Abrir un archivo'
)

# Definir un delimitador de archivo
conjunto = np.genfromtxt(archivo, delimiter=',')


# # Dataset de iris
# - https://archive.ics.uci.edu/ml/machine-learning-databases/iris/   # iris.data
# - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html # sklean.dataset load_iris().

# In[90]:


# Configuracion del modelo
# P atributos a tomar (-1 para todos los atributos), X atributos, Y salidas
# Regularmente las salidas es la ultima columna (-1)
p = 2
X = conjunto[:, :p]
Y = conjunto[:, -1]
Y = Y[:, np.newaxis]
[ren, col] = conjunto.shape

# Plotear los datos acorde el numero de salidas (Y) distintas
uni = np.unique(Y).astype(int)

# Mapa de colores generado al azar
colores = np.random.random_sample((len(uni), 3))
colores_t = tuple(map(tuple, colores))

for i in range(len(uni)):
    plt.scatter(X[Y[:, 0] == uni[i], 0], X[Y[:, 0] == uni[i], 1], 70, color=colores_t[i], marker='+')
    
plt.show()


# # Sin el for, se haría a mano de la siguiente manera.
# - Se tendria que definir cada salida con un color distinto y un numero diferente.
#     plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue") # Iris-setoza = 0
#     plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")  # Iris-versicolor = 1
#     plt.scatter(X[Y[:, 0] == 2, 0], X[Y[:, 0] == 2, 1], c="olivedrab") # Iris-verginica = 2

# In[91]:


# Clase para definir capas de la red
class neural_layer():
    """ n_conn = Numero de conexiones
        n_neur = Numero de neuronas
        act_f = Funcion de activacion
        b = Bias o sezgo
        W = Pesos """
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1


# In[92]:


# Funciones de activacion
# Sigmoide + derivada de la funcion sigmoide
sigm = (lambda x: 1 / (1 + np.e ** (-x)),
       lambda x: x * (1 - x))

# Relux
relu = lambda x: np.maximum(0, x)

# Test de las funciones
#_xt = np.linspace(-5, 5, 100)
#plt.plot(_xt, relu(_xt))
#plt.plot(_xt, sigm[0](_xt))


# In[93]:


# Crear la red neuronal
# El numero de capas inicial se representa por P, el total de neuronas a usar por capa y la funcion de activacion.
# Una topologia de red en IA deriva a la mulplicacion de neuronas de la primera capa al doble.
def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))
    
    return nn


# In[97]:


# FUNCION DE ENTRENAMIENTO

topology = [p, 4, 8, 1]

neural_net = create_nn(topology, sigm)  

l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))



def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
  
  out = [(None, X)]
  
  # Forward pass
  for l, layer in enumerate(neural_net):
  
    z = out[-1][1] @ neural_net[l].W + neural_net[l].b
    a = neural_net[l].act_f[0](z)
  
    out.append((z, a))
    
  
  if train:
    
    # Backward pass 
    deltas = []
    
    for l in reversed(range(0, len(neural_net))):
      
      z = out[l+1][0]
      a = out[l+1][1]
      
      if l == len(neural_net) - 1:
        deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
      else:
        deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))
       
      _W = neural_net[l].W
 
      # Gradient descent
      neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr   
      neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr
      
  return out[-1][1]

# In[99]:

# Probar la red
for i in range(1000):
    costo = train(neural_net, X, Y, l2_cost, 0.5)
    print(costo)

