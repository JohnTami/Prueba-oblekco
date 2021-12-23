# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:02:21 2021

@author: John Tami
"""

""" Lectura de los datos de entrenamiento y de test """
import pandas as pd

train  = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%%
""" Limpieza de los Tweets """

import numpy as np 
import re

# Función para limpieza de datos
def eliminar_patrones(tex_original, patron):
    r = re.findall(patron, tex_original)
    for i in r:
        tex_original = re.sub(i, '', tex_original)
    return tex_original 

# Eliminar palabras que inicien con @ (@usuarios)
train['tweet_limpio'] = np.vectorize(eliminar_patrones)(train['tweet'], "@[\w]*")
test['tweet_limpio'] = np.vectorize(eliminar_patrones)(test['tweet'], "@[\w]*")

# Eliminar números, puntuaciones y caracteres especiales
train['tweet_limpio'] = train['tweet_limpio'].str.replace("[^a-zA-Z#]", " ")
test['tweet_limpio'] = test['tweet_limpio'].str.replace("[^a-zA-Z#]", " ")

# Eliminar palabras con menos de 4 letras de longitud
train['tweet_limpio'] = train['tweet_limpio'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
test['tweet_limpio'] = test['tweet_limpio'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
 

#%%
""""Tokenización"""

tokenized_tweet_train = train['tweet_limpio'].apply(lambda x: x.split())
tokenized_tweet_test = test['tweet_limpio'].apply(lambda x: x.split())


from nltk.stem.porter import *
stemmer = PorterStemmer()
 
tokenized_tweet_train = tokenized_tweet_train.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet_test = tokenized_tweet_test.apply(lambda x: [stemmer.stem(i) for i in x])

#%%

"""                    Nube de  palabras - Palabras negativas              """

import matplotlib.pyplot as plt 
from wordcloud import WordCloud

negative_words = ' '.join([text for text in train['tweet_limpio'][train['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#%%
""" Extraer hashtag dependiendo de si es o no racista o sexista"""

# Función para extraer los hashtags
def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags
 
# Extraer HT de Tweets no racistas, ni sexistas
HT_normal = hashtag_extract(train['tweet_limpio'][train['label'] == 0])
 
# Extraer HT de Tweets racistas y sexistas
HT_negativo = hashtag_extract(train['tweet_limpio'][train['label'] == 1])


#%%

""""Entrenamiento para clasificación de tweets"""

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
train_bow = bow_vectorizer.fit_transform(train['tweet_limpio'])
test_bow = bow_vectorizer.fit_transform(test['tweet_limpio'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
 

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

#%%
"""Regresión Logística"""

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
 
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
 
f1_score(yvalid, prediction_int) # calculating f1 score

#%%
"""Predicción y exportación de los datos"""

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
result = test[['id','tweet','label']]
result.to_csv('prediccion.csv', index=False) # writing data to a CSV file
