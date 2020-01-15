#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 1 01:04:45 2020

@author: JMontiel
"""
# In[2]
import numpy as np
import matplotlib.pyplot as plt

# In[3]
# Conjunto de datos de la compuerta logica AND
conjunto = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0],  [1, 1, 1]])

# In[4]
# Configuracion del algoritmo
r, c = conjunto.shape
# S entradas, R salidas.
p = conjunto[:, :c-1].transpose()
s, k = p.shape
t = conjunto[:, -1].reshape(r, 1)
o, r = t.shape
bias = np.random.rand(r, 1)
pesos_w = np.random.rand(r, s)
iteraciones = 50
contador = 0
error = np.zeros((r, k))

# In[5]
# Funcion hardlimit
# Resive un valor como el ejemplo de matlab, evalua comparando y retorna 0 o 1.
def hardlim(entrada):
    return (entrada > 0) * 1.0

# In[6]
# Ejecucion del algoritmo
while(contador < iteraciones):
    for i in range(k):
        # @ Se utiliza para multiplicacion de matrices
        n = pesos_w @ p[:, i] + bias
        
        a = hardlim(n)
        
        error[:, i] = t[i] - a.transpose()
        
        if abs(error[:, i] != 0):
            pesos_w = pesos_w + (error[:, i] * p[:, i].transpose())
            bias = bias + error[:, i]
            
    
    error_total = sum(sum(abs(error)))
    
    print('Iteracion ', contador+1, ' de ', iteraciones, ' Error total: ', error_total)

    if error_total == 0:
        break
    contador = contador + 1

# In[7]
#plt.scatter(p[0, :], p[1, :])
#plt.show()
# Funcion para plotear
def plot_data(inputs,targets,weights):
    # fig config
    plt.figure(figsize=(10,6))
    plt.grid(True)

    #plotear entradas de 2 dimensiones con 2 clases 0 y 1.
    #Plotear los puntos por colores
    for input,target in zip(inputs,targets):
        plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')

    # Calcular la intercepcion
    for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
        pendiente = -(weights[0, 0] / weights[0, 1])
        intercepcion = -weights[0, 0] / weights[0, 1]
        #slope = -(weights[0]/weights[2])/(weights[0]/weights[1])  
        #intercept = -weights[0]/weights[2]

        #y =mx+c, m is slope and c is intercept
        y = (pendiente*i) - intercepcion
        plt.plot(i, y,'ko')

# In[8]
plot_data(p.transpose(), t, pesos_w)