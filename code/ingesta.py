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
from string import punctuation
import itertools
from random import shuffle
from nltk.corpus import stopwords
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
"""
documento trae un file abierto
lo consumo a memoria con read() a un solo string
lo parto segun keyterms = Honorable asamblea
"""
for documento in documentos:
    documento_nombre = documento.name.rsplit('/',1)[1].split('.')[0]
    raw = documento.read()
    for i,tematica in enumerate(raw.replace("Honorable Asamblea:","Honorable asamblea:").split("Honorable asamblea:")):
        with io.open(path_to_raw+'tematicas/'+documento_nombre+'_'+str(i+1)+'.txt', mode="w") as newfile:
            tematica_limpia = mataAcentos(strip_punctuation(tematica.replace('\n',' ')))
            newfile.write(tematica_limpia)
            
for documento in documentos:
    documento_nombre = documento.name.rsplit('/',1)[1].split('.')[0]
    raw = documento.read()
    for i,tematica in enumerate(raw.replace("Honorable Asamblea:","Honorable asamblea:").split("Honorable asamblea:")):
        with io.open(path_to_raw+'tematicas_to_rake/'+documento_nombre+'_'+str(i+1)+'.txt', mode="w") as newfile:
            tematica_limpia = toLowerCase(tematica)
            newfile.write(tematica_limpia)

def toLowerCase(s):
    data = s.lower()
    return data

def mataAcentos(s):
    charstosub = pd.DataFrame(zip([u'á', u'é', u'í', u'ó', u'ú',u'"',u'“',u'”',u',',u'\.',u'ñ',u'\!',u'\¡'],[u'a', u'e', u'i', u'o', u'u',u'',u'',u'',u'',u'',u'n',u'\.',u'\.'])) 
    data = s.lower()
    for row in charstosub.iterrows():
        data = re.sub(row[1][0],row[1][1],data)
    return data
##Este no funciona aun
"""
Remover acentos y puntuacion parece no ser lo mejor


def cleanText(corpus):
    #punctuation = """.,?!:;(){}[]"""
    punctuation = punctuation
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus
"""

#Esto si funciona
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
"""
Christian!!!!!
Crea una carpeta prueba y mete alli documentos de prueba para RAKE
"""

############################################ Doc2Vec

file_names_tematicas = [f for f in os.listdir(path_to_raw+'tematicas/') if f.endswith('.txt')]
sentences = []
i=0
dicto = {}
for doc in file_names_tematicas:
    f = io.open(path_to_raw+'tematicas/'+doc,'rt')
    sentences.append(gensim.models.doc2vec.LabeledSentence(f.read().split(),[doc]))
    f.close()
    
    
#Pruebas
sentences[0]
sentences[0:5]
print(sentences[0:5])
len(sentences)


#Modelo
min_count = 5
size = 300
window = 10
 
model = gensim.models.doc2vec.Doc2Vec(sentences,size = size, window = window, min_count = min_count)


#################Algunas Pruebas
dir(model)
len(model.vocab)
model.similarity('organismos','constar')
checa(u'elecciones')
def checa(word):
    for i in model.most_similar(word):
        print(i)

print model.docvecs.most_similar(["dd_1970_8.txt"])
model.docvecs["dd_1970_8.txt"]
dir(model.docvecs)
model.docvecs.doctags
dir(model)

############################Armar datos para clustering
mylist = []
tags=[]
for doc in model.docvecs.doctags:
    tags.append(doc)    
    mylist.append(model.docvecs[doc])

model.docvecs.doctags
tags[0]
mat = np.array(mylist)
############################################### KMeans

for i in range(10):
    agrupamiento = KMeans(n_clusters = i+1)
    agrupamiento.fit(mat)
#    mylist['clasificacion_'] = agrupamiento.labels_.tolist()
    
from scipy.cluster.vq import kmeans,vq
K = range(1,10)

KM = [kmeans(mat,k) for k in K]
centroids = [cent for (cent,var) in KM] 
avgWithinSS = [var for (cent,var) in KM] # suma de cuadrados promedio intra-cluster 

##### plot ###
kIdx = 4

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Numero de clusters')
plt.ylabel('Suma de cuadrados promedio intra-cluster')
plt.title('Criterio del codo para K-Medias')

plt.show()


# KMeans con el resultado del criterio del codo
agrupamiento = KMeans(n_clusters = 5)
agrupamiento.fit(mat)
agrupamiento.labels_


agrupados = pd.DataFrame(zip(tags, agrupamiento.labels_.tolist()))

n_clusters=5


lista_cluster =  [[] for i in range(n_clusters)]
for i in range(len(agrupados)):
   lista_cluster[agrupados[1][i]].append(agrupados[0][i])
   
for i in range(len(lista_cluster)):
    print("Clster"+str(i),len(lista_cluster[i]))

for i in range(len(lista_cluster)):
    with io.open(path_to_raw+'clusters_to_rake/'+"cluster_"+str(i)+".txt", 'w') as outfile:
        filenames = lista_cluster[i]
        for fname in filenames:
            with io.open(path_to_raw+'tematicas_to_rake/'+fname) as infile:
                for line in infile:
                    outfile.write(line)

######################################MajorClust
#### Calculo de Cosine Similarity
num_of_samples, num_of_features = mat.shape

magnitudes = np.zeros((num_of_samples))
# this loop can be removed?
for doc_id in range(num_of_samples):
  magnitudes[doc_id] = np.sqrt(mat[doc_id].dot(mat[doc_id].T).sum())

cosine_distances = np.zeros((num_of_samples, num_of_samples))
# this loop can be improved
for doc_id, other_id in itertools.combinations(range(num_of_samples), 2):
  distance = (mat[doc_id].dot(mat[other_id].T).sum())/(magnitudes[doc_id]*magnitudes[other_id])
  cosine_distances[doc_id, other_id] = cosine_distances[other_id, doc_id] = distance
 
##### MajorClust Implementation 
finish = False
iters = 1
shuffled_indices = np.arange(num_of_samples)
shuffle(shuffled_indices)
while not finish:
  finish = True
  for index in shuffled_indices:
    # aggregating edge weights  
    # bincount cuanta el numero de ocurrencias de un valor en un arreglo y lo pesa por weights
    new_index = np.argmax(np.bincount(shuffled_indices,weights=cosine_distances[index]))
    if shuffled_indices[new_index] != shuffled_indices[index]:
      shuffled_indices[index] = shuffled_indices[new_index]
      finish = False

bins = pd.DataFrame(zip(np.arange(num_of_samples),np.bincount(shuffled_indices)))
bins[bins[1] != 0].sort_values(1,ascending=False)



clusters = {}
for item, index in enumerate(indices):
  clusters.setdefault(index, []).append(links[-50+item]["url"])

######################### RAKE ############################################

print(nltk.corpus.stopwords.words('spanish'))


