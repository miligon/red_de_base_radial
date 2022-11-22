#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 20:17:11 2022

@author: miguel
"""

import random
import numpy as np

# Lee la informacion desde un archivo CSV de 4 columnas
def load_data(ruta, n_muestras_train = 50):
    attrib_e = [[],[],[],[]]
    labels_e = []
    
    attrib_p = [[],[],[],[]]
    labels_p = []
    
    #Leer datos
    a = open(ruta)
    buffer = list()
    for l in a:
        buffer.append(l.strip())
    
    #Reordenar aleatoriamente    
    random.shuffle(buffer)
    
    for i in range(len(buffer)):
        if (i < n_muestras_train):
            #print("entrenamiento ", i, buffer[i])
            sepal_l, sepal_w, petal_l, petal_w, label = buffer[i].split(",")
            attrib_e[0].append(float(sepal_l))
            attrib_e[1].append(float(sepal_w))
            attrib_e[2].append(float(petal_l))
            attrib_e[3].append(float(petal_w))
            labels_e.append([int(i),label])
        else:
            #print("pruebas", i, buffer[i])
            sepal_l, sepal_w, petal_l, petal_w, label = buffer[i].split(",")
            attrib_p[0].append(float(sepal_l))
            attrib_p[1].append(float(sepal_w))
            attrib_p[2].append(float(petal_l))
            attrib_p[3].append(float(petal_w))
            labels_p.append([int(i-n_muestras_train),label])            
    
    return np.array(attrib_e), np.array(labels_e), np.array(attrib_p), np.array(labels_p)


# Carga la información del modelo de bicicletas: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
# Atributos: columnas 2-10
# Valor a predecir: numero de bicicletas usadas por día (demanda)
def load_data_bike(ruta, n_muestras_train = 50):
    attrib_e = [[],[],[],[],[],[],[],[],[],[],[]]
    labels_e = []
    
    attrib_p = [[],[],[],[],[],[],[],[],[],[],[]]
    labels_p = []
    
    #Leer datos
    a = open(ruta)
    buffer = list()
    for l in a:
        buffer.append(l.strip())
        
    
    #Reordenar aleatoriamente    
    #random.shuffle(buffer)
    
    for i in range(2,len(buffer)):
        if (i < n_muestras_train):
            #print("entrenamiento ", i, buffer[i])
            columnas = buffer[i].split(",")
            label = columnas[15]
            for j in range(2,13):
                attrib_e[j-2].append(float(columnas[j]))
            labels_e.append([int(i),label])
        else:
            #print("pruebas", i, buffer[i])
            columnas = buffer[i].split(",")
            label = columnas[15]
            for j in range(2,13):
                attrib_p[j-2].append(float(columnas[j]))
            labels_p.append([int(i-n_muestras_train),label])          
    
    return np.array(attrib_e), np.array(labels_e), np.array(attrib_p), np.array(labels_p), None