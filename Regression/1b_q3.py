#!/usr/bin/env python
# coding: utf-8

# In[58]:


import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# In[59]:


df = pd.read_csv("data/admission_predict.csv")
df = df.drop(['Serial No.'], axis=1)

features = df.drop(["Chance of Admit"], axis=1)
y = df['Chance of Admit']


# In[60]:


#no of features
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]

#Stores feature ranking from rfe. 
rfe_ranking = []
for n in range(5, 7):
    X_train, X_test, y_train, y_test = train_test_split(features,y, test_size = 0.3, random_state = 42)
    model = LinearRegression()
    rfe = RFE(model,n) #second arg is number of features to select
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    
    
    rfe_ranking.append(rfe.ranking_)
    
    if(score>high_score):
        high_score = score
        nof = n
        
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[61]:


print(rfe_ranking[0])
print(rfe_ranking[1])


# In[62]:


#5 features selected
        
print("5 Features Selected with RFE:")
    
for i in range (len(rfe_ranking[0])):
    #selected feature
    if (rfe_ranking[0][i] == 1): 
        print(df.columns[i])

print("\nDropped features:")        
for i in range (len(rfe_ranking[0])):
    if(rfe_ranking[0][i] > 1):
        print(df.columns[i])
        

        
print("\n#########\n")
print("6 Features Selected with RFE:")
#6 features selected
for i in range (len(rfe_ranking[1])):
    #selected feature
    if (rfe_ranking[1][i] == 1): 
        print(df.columns[i])
        
print("\nDropped features:")        
for i in range (len(rfe_ranking[1])):
    if(rfe_ranking[1][i] > 1):
        print(df.columns[i])


# In[ ]:




