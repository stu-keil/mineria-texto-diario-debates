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

os.getcwd()
os.chdir("/home/stuka/itam2/textmining/mineria-texto-diario-debates/data/raw/")
os.listdir(".")

path_to_raw = "/home/stuka/itam2/textmining/mineria-texto-diario-debates/data/raw/"

file_names = [f for f in os.listdir(path_to_raw) if f.endswith('.txt')]

documentos =[io.open(f,'rt',encoding='ISO-8859-1') for f in file_names]


raw = data[0].read()

documentos[1]

dir(data[0])
data[0].name

len(documentos[0].read(20000).split("Honorable asamblea:"))

for documento in documentos:
    documento_nombre = documento.name.split(".")[0]
    raw = documento.read()
    tematicas = raw.split("Honorable asamblea:")
    for tematica in range(len(documentos)):
        contenido = tematicas[tematica]
        documento_nombre+'_'tematica
        
        
        

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield gensim.models.doc2vec.LabeledSentence(words=line.split(), labels=['TXT_%s' % uid])


sentences = LabeledLineSentence(documentos[0])
for line in sentences:
    print(line)

lines = data[0].readlines()
listota [] 
for line in lines:
    print(len(line.split(".")))
    
line = lines[0]