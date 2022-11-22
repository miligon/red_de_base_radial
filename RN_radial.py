#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:25:39 2022

@author: miguel
"""

from kmeans import Kmeans 
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class perceptron:
    def __init__(self, entradas, w = 0, umbral = 0, activacion = "none"):
        if w == 0 :
            #self.w = (np.random.rand(1, entradas) * 2) - 1
            #self.w = np.tile(1/entradas, entradas)
            self.w = np.tile(0, entradas)
        else:
            self.w = w
        
        if umbral == 0 :
            self.umbral = 0
        else:
            self.umbral = umbral
            
        if activacion == "tanh" :
            self.f_activacion = lambda x: math.tanh(x)
        elif (activacion == "sigm"):
            self.f_activacion = lambda x: 1 / (1 + np.exp(-x))
        else:
            self.f_activacion = lambda x: x
        self.entradas = entradas
    
    def calcSalida(self, data, modo = "clasificacion"):
        y = np.sum(data * self.w.T) - self.umbral  
        # Funcion de activación tanh
        y = self.f_activacion(y)
        # Funcion de activación sigmoide
        #y = 1/(1+math.exp(-y))
        if modo == "clasificacion":
            if y > 0:
                y = 1
            elif y <= 0:
                y = -1
        return y


class RNA:
    
    def __init__(self, numNeuronas=2, n_salidas = 1):
        self.n_neuronas = numNeuronas
        self.capa_salida = [perceptron(numNeuronas) for i in range(n_salidas)]
        return
    
    def __calcCentroid(self, data):
        clasificador = Kmeans()
        centroids, groups, varianzas = clasificador.run(data,self.n_neuronas)
            
        self.centroids = centroids
        self.varianzas = varianzas
    
    def __calcFi(self, data):
        phiArray = []
        dataSet = data
        #print(dataSet.shape, len(self.centroids[0,:]), self.centroids.shape)
        
        for j in range(self.n_neuronas):
            # Calculo de distancias al centroide
            centroid = np.tile(self.centroids[j,:],(dataSet.shape[0],1))
                        
            diff = (dataSet - centroid)**2
            dist = np.sum( diff, axis=1)
            # Funcion de base radial
            phi = np.exp((-1*(dist))/(2*self.varianzas[j])).tolist()
            phiArray.append(phi)
        return np.array(phiArray)
    
    def runHebb(self, trainSet, labels, f_aprendizaje, epocas=5000, modo = "clasificacion"):
        phis = self.__calcFi(trainSet)
        labels = labels[:,1]
        error_flag = True
        epoch_e_count = 0
        last_w = []
        epoch = 0
        while (error_flag and epoch < epocas):
            error_flag = False
            bad = 0
            salidasArray = []
            for index, element in enumerate(phis.T):
                salidas = []
                for i_n, neurona in enumerate(self.capa_salida):
                    #print(element.shape)
                    salida = neurona.calcSalida(element, modo)
                    salidas.append(salida)
                    entradas = np.array(element)
                    
                    if labels[index] != salida:
                        bad +=1
                        error_flag = True
                        error = labels[index] - salida
                        neurona.w = neurona.w + (error * entradas * f_aprendizaje)
                        
                        if (epoch < 100):
                            print(epoch, neurona.w, (error * entradas * f_aprendizaje))
                    #input()
                        
                    #exit()
                salidasArray.append(salidas)
            epoch += 1
            print ("Epocas: {}/{}".format(epoch, epocas) ,end="")
            print(" | Error total del aprendizaje: ", round(bad/len(labels)*100,2), "%",end="\r")
           
        
        print ("Epocas: {}/{}".format(epoch, epocas), end="")
        
        bad = 0
        for index, element in enumerate(salidasArray):
            if labels[index] != element:
                bad +=1
        
        print("Error total del aprendizaje: ", round(bad/len(labels)*100,2), "%")
            
        
    def runNormalEquations(self, trainSet, labels):
        phis = self.__calcFi(trainSet).T
        print("matriz de phis: ", phis.shape)
        labels = np.array([labels[:,1]]).T
        print("matriz de salidas deseadas: ", labels.shape)
        inversa = np.linalg.pinv(phis)
        print("matriz de phis inversa: ", inversa.shape)
        w = np.dot(inversa, labels).T
        print("matriz de pesos: ", w.shape)
        
        for index, pesos in enumerate(w):
            for i_n, neurona in enumerate(self.capa_salida):
                neurona.w = pesos
                print(pesos)
        
        print("Pesos calculados!")
        return
        
        
    def train(self, trainSet, labels, f_aprendizaje= 0.5, epocas = 5000, train_mode = "hebb", modo = "clasificacion"):
        trainSet = np.array(trainSet).T
        labels = np.array(labels)
        unique_labels = np.unique(labels[:,1], axis=0)
        self.__calcCentroid(trainSet)
        if train_mode == "hebb":
            self.runHebb(trainSet, labels, f_aprendizaje,epocas, modo)
        elif (train_mode == "pinv"):
            self.runNormalEquations(trainSet, labels)            
        
    def classify(self, data):
        data = np.array([data])
        phis = self.__calcFi(data)
        element = phis.T[0,:]
        salidas = []
        for i_n, neurona in enumerate(self.capa_salida):
            salida = neurona.calcSalida(element)
            salidas.append(salida)
        return salidas
        

#***************************************************

# #Valores de Prueba
# n = 100
# g = 4
# k = 2
# width = random.random() * 100

# x = np.random.randint(low=1, high=500, size=(1,n))
# y = np.random.randint(low=1, high=500, size=(1,n))
# z = np.random.randint(low=1, high=500, size=(1,n))

# #figure, axes = plt.subplots()
# #axes = plt.scatter(x,y)
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
# ax.scatter3D(x, y, z, color = "green")

# attrib = np.vstack((x,y,z))

# z = np.array([[i] for i in range(k) ])
# labels = np.repeat(z, int(x.shape[1])/k)
# labels = [[i, labels[i]] for i in range(len(labels))]

# rna = RNA()
# rna.train(attrib, labels )
