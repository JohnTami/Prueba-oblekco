# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 10:58:18 2021

@author: John Tami
"""

##############################################################################
#                           Ejercicio práctico
##############################################################################

""" En el correo, se adjunta un archivo JSON con tweets. Debes leer el archivo 
en Python y obtener:"""

##############################################################################
#                      Lectura de datos formato json
##############################################################################

import json

with open('tweets_test.json') as file:
    data = json.load(file)

##############################################################################
# 1. Número de tweets
print("El número de Tweets es", len(data))

#%%
##############################################################################
# 2. Lista de los usuarios autores de los tweets 
#(tip: el nombre de usuario está en user > screen_name).

Usuarios=[]
for n in data.keys():
    Usuarios.append(data[n]["user"]["screen_name"])
    #print(data[n]["user"]["screen_name"])
Usuarios1=set(Usuarios)  
print("Lista de usuarios:\n","\n".join(Usuarios1))

#%%
##############################################################################
# 3. Totalizador del numero de seguidores de todos los usuarios 
# (tip: los seguidores están en user > followers_count)

Total_seguidores=0
for n in data.keys():
    Total_seguidores=Total_seguidores + data[n]["user"]["followers_count"]
print("El número de seguidores de todos los usuarios es ",Total_seguidores)

#%%
##############################################################################
# 4. Obtén una lista de todos los usuarios mencionados en todos los tweets

Usuarios_menc=[]
for n in data.keys():
    for i in data[n]["real_entities"]["user_mentions"]:
        Usuarios_menc.append(i["screen_name"])
        #print(data[n]["user"]["screen_name"])
Usuarios_menc1=set(Usuarios_menc)  
print("Lista de usuarios mencionados en todos los tweets:\n","\n".join(Usuarios1))

#%%
##############################################################################
# 5. Obtén el total de tweets con tendencia 0, 1 y 2 respectivamente.

Tend_0=0
Tend_1=0
Tend_2=0
for n in data.keys():
    print(n)
    if data[n]["tendencia"] == "0":
        Tend_0=Tend_0+1
    elif data[n]["tendencia"] == "1":
        Tend_1=Tend_1+1
    elif data[n]["tendencia"] == "2":
        Tend_2=Tend_2+1
        
print("Total de Tweets con tendencia 0 es: ",Tend_0)
print("Total de Tweets con tendencia 1 es: ",Tend_1)
print("Total de Tweets con tendencia 2 es: ",Tend_2)
