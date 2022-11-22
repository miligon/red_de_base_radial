#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:10:46 2022

@author: miguel
"""

from knn import KNN
from RN_radial import RNA
from eigenfaces import EIGENFACES
import numpy as np
import imagenes
import statistics

# ---------------     Carga de datos     --------------------
# Función que carga en memoria el conjunto de datos de caras fei
# el conjunto debe estar etiquetado correctamente. Si el nombre de la imagen contiene h, el algoritmo lo identifica como hombre
# si el nombre de la imagen contiene m, el algoritmo lo identifica como mujer. Ej. 1ah,jpg -> hombre, 1bm.jpg -> mujer
def load_fei(n_test):
    dataset_path = "fei/frontalimages_manuallyaligned_part1/"
    n_training = 200 - n_test
    print("# datos de entrenamiento:", n_training, ", # de datos prueba:", n_test)
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels = imagenes.load_data(
        dataset_path, n_training
    )
    return train_attrib, train_labels, test_attrib, test_labels, n_test

# Ejecuta EIgenfaces para clasificacion usando el conjunto de prueba test_attrib, test_labels, n_testing
def rna_clasify(rna, test_attrib, test_labels, n_testing):
    good = 0
    bad = 0
    #print(train_attrib, train_labels)
    for a in range(len(test_attrib[0])):
        #print(test_attrib[:,a].shape)
        res = rna.classify(test_attrib[:,a].T)
        index, name = test_labels[a]
        
        if (name == res[0]):
            good = good + 1
            #print("Esperado:", name, " Obtenido:", res[0], "OK")
        else:
            bad = bad + 1
            #print("Esperado:", name, " Obtenido:", res[0], "FALLÓ")
            
    good_p = round((good/n_testing)*100,2)
    bad_p = round((bad/n_testing)*100,2)
    print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing)
    return [good_p, bad_p]

def ejecuta_rna_fei(n_training):
    # Importa datos
    train_attrib, train_labels, test_attrib, test_labels, n_testing = load_fei(n_training)
    #print(train_attrib, train_labels)
    n_caracteristicas = len(train_attrib[0])
    
    # Creación de la instancia de la red neuronal
    # Neuronas de la capa oculta, Neuronas de la capa de salida
    rna = RNA(60, 1)
    # Entrenamiento usando Hebb
    #rna.train(train_attrib, train_labels, 130, 5000, train_mode = "hebb")
    
    # Entrenamiento usando pseudoinversa
    rna.train(train_attrib, train_labels, None, None, train_mode = "pinv")
    
    return rna_clasify(rna, test_attrib, test_labels, n_testing)
    

# Funcion principal
if __name__ == '__main__':
    n_iter = 10
    e_good = []
    e_bad = []
    for i in range(n_iter):
        print("\n\nRonda de prueba: {}/{}".format(i+1,n_iter))
        # Define la longitud del conjunto de prueba
        n_test =50
        
        # Seleccion de prueba a ejecutar
        
        # Clasificación:
        good, bad = ejecuta_rna_fei(n_test)
            
        e_good.append(good)
        e_bad.append(bad)
    
    prom_e_good = round(statistics.mean(e_good),2)
    prom_e_bad = round(statistics.mean(e_bad),2)
    
    dev_std_good = round(statistics.stdev(e_good),2)
    dev_std_bad =  round(statistics.stdev(e_bad),2)
    
    print("promedio Good: ", prom_e_good, "% , promedio bad: ", prom_e_bad, "%, # de iteraciones: ", n_iter)
    print("dev std: ", dev_std_good, "% , promedio bad: ", dev_std_bad, "%")