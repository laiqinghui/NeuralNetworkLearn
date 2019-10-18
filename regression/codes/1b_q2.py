#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


# In[34]:


df = pd.read_csv("data/admission_predict.csv")
df = df.drop(['Serial No.'], axis=1)


# In[35]:


corr_matrix = df.corr(method='pearson')


# In[36]:


plt.matshow(corr_matrix)
plt.colorbar()
plt.show()


# In[37]:


corr_matrix


# In[38]:


temp_corr_matrix = corr_matrix.values
np.fill_diagonal(corr_matrix.values, 0)

corr_row_sum = []

for i in range (temp_corr_matrix.shape[0]):
    corr_row_sum.append(np.sum(temp_corr_matrix[i]))

    
print (corr_row_sum)


# In[39]:


np.fill_diagonal(corr_matrix.values, np.nan)

order_top2 = np.argsort(-corr_matrix.values, axis=1)[:, :6]
order_bottom = np.argsort(corr_matrix.values, axis=1)[:, :1]

result_top2 = pd.DataFrame(
    corr_matrix.columns[order_top2], 
    columns=['1st', '2nd', '3rd', '4th', '5th', '6th'],
    index=corr_matrix.index
)

result_bottom = pd.DataFrame(
    corr_matrix.columns[order_bottom], 
    columns=['Last'],
    index=corr_matrix.index
)

result = result_top2.join(result_bottom)

for x in result.columns:
    result[x+"_Val"] = corr_matrix.lookup(corr_matrix.index, result[x])
    


# In[40]:



result.insert(0, "Corr Sum", corr_row_sum)
result

