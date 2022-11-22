#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:18:53 2022

@author: miguel
"""

#Loading the required modules
 
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
import time

class Kmeans:

    def _get_distance(self, centroides, atributos):
        distancias = []
        varianzas = []
        centroids = centroides
        for i in centroids:
            #exit()
            a = np.tile(i, (len(atributos[0]),1))
            b = atributos
            c = b-i
            c = np.power(c, 2)
            c = np.sum(c, axis=1)
            c = np.sqrt(c)
            distancias.append(c.tolist())
            varianzas.append(statistics.variance(c.tolist()))
            #varianzas.append(statistics.mean(c.tolist()))
        return distancias, varianzas
     
    def run(self, data, k, max_iterations = 1000):
        data = np.array(data)
        longitud = data.shape[0]
        last_centroids = []
        counter = 0
        
        # Centroides
        indices = np.random.choice(longitud, k,replace=False)
        
        centroides = []
        for i in indices:
            centroides.append(data[i,:])
        centroides = np.array(centroides)
        
        # Distancias Puntos - centroides
        distances, varianzas = self._get_distance(centroides, data)
        distances = np.array(distances)
        
        # etiquetar puntos en base al centroide mas cercano
        puntos_etiquetados = np.array([np.argmin(distances[:,i]) for i in range(len(distances[0]))])
        reselect_centroids = False
        
        for iters in range(max_iterations): 
            centroides = []
            for j in range(k):
                centroid = []
                for i in range(data.shape[1]):
                    #print(data.shape, len(data[0]), puntos_etiquetados.shape)
                    #exit()
                    # Promediar los valores de los puntos asociados a un centroide
                    tmp = [ data[x][i] for x in range(data.shape[0]) if puntos_etiquetados[x] == j ]
                    if ( len(tmp) == 0):
                        # En caso de que el arreglo para un centroide este vacio, reasignar aleatoriamente los centroides :(
                        reselect_centroids = True
                    else:
                        centroid.append(statistics.mean(tmp))
                    
                centroides.append(centroid)
                
            if reselect_centroids :
                indices = np.random.choice(longitud, k,replace=False)
                centroides = []
                for i in indices:
                    centroides.append(data[i,:])
                centroides = np.array(centroides)
                reselect_centroids = False
                
            # Recalcular distancias y etiquetar puntos
            distances, varianzas = self._get_distance(centroides, data)
            distances = np.array(distances)
            puntos_etiquetados = np.array([np.argmin(distances[:,i]) for i in range(len(distances[0]))])
            
            # Si el valor de los centroides no cambia en 10 iteraciones seguidas -> salir del ciclo for
            if last_centroids == centroides:
                counter += 1
                if (counter >= 2):
                    break
            else:
                last_centroids = centroides
                counter = 0
            print("K-means: {}/{}".format(iters, max_iterations), end="\r")
        
        print("K-means: {}/{} completado!".format(iters, max_iterations))
        grupos = []
        caracteristicas = []
        for j in range(k):
            caracteristicas = []
            for i in range(data.shape[1]):
                caracteristicas.append([ data[x][i] for x in range(data.shape[0]) if puntos_etiquetados[x] == j ])
            grupos.append(caracteristicas)
        c = np.array(centroides)
        return c, grupos, varianzas
 