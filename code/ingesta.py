# -*- coding: utf-8 -*-
"""
Created on Thu May 12 23:24:54 2016

@author: stuka
"""

import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import os
import io
import re
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import gensim
#Cleaning the house
os.getcwd()
os.chdir("/home/stuka/itam2/textmining/mineria-texto-diario-debates/data/raw/")
os.listdir(".")

#Debe venir de fuera
path_to_raw = "/home/stuka/itam2/textmining/mineria-texto-diario-debates/data/"

file_names = [f for f in os.listdir(path_to_raw+'raw/') if f.endswith('.txt')]

documentos =[io.open(f,'rt',encoding='ISO-8859-1') for f in file_names]


for file_name in file_names:
    sourceEncoding = "iso-8859-1"
    targetEncoding = "utf-8"
    source = io.open(file_name,'rt',encoding='ISO-8859-1')
    target = open(path_to_raw+'encoded/'+file_name, "w")
    target.write(source.read().encode(targetEncoding))
    
documentos =[io.open(path_to_raw+'encoded/'+f,'rt') for f in file_names]

#Pruebas
raw = documentos[0].read()

documentos[1]

dir(data[0])
data[0].name
#Codigo para probar stopterm
for documento in documentos:
    print(len(documento.read().replace("Honorable Asamblea:","Honorable asamblea:").split("Honorable asamblea:")))
    
    
#Este es el buen codigo
for documento in documentos:
    documento_nombre = documento.name.rsplit('/',1)[1].split('.')[0]
    raw = documento.read()
    for i,tematica in enumerate(raw.replace("Honorable Asamblea:","Honorable asamblea:").split("Honorable asamblea:")):
        with io.open(path_to_raw+'tematicas/'+documento_nombre+'_'+str(i+1)+'.txt', mode="w") as newfile:
            newfile.write(tematica.replace('\n',' '))
 
       
file_names_tematicas = [f for f in os.listdir(path_to_raw+'tematicas/') if f.endswith('.txt')]
sentences = []
i=0
for doc in file_names_tematicas:
    f = io.open(path_to_raw+'tematicas/'+doc,'rt')
    sentences.append(gensim.models.doc2vec.LabeledSentence(f.read().split(),["SENT_"+str(i+1)]))
    i=i+1
    f.close()

sentences[0]
sentences[0:5]
print(sentences[0:5]) 

test = sentences[0:5]

min_count = 2
size = 50
window = 4
 
model = gensim.models.doc2vec.Doc2Vec(sentences,size = size, window = window, min_count = min_count)
dir(model)
model.vocab
model.similarity('organismos','constar')
model.most_similar('hombre')
print model.docvecs.most_similar(["SENT_1"])
model.docvecs["SENT_1"]
