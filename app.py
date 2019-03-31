#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


# In[2]:


df = pd.read_csv("data/training1.csv" , delim_whitespace=True, header=None)
df.head()


# In[3]:


features =[
    "date",
    "TaipeiTemp",
    "TaoyuanTemp",
    "TaichungTemp",
    "TainanTemp",
    "KaohsingTemp",
    "vacation"
]

target= "peak_load(MW)"


# In[4]:


targets=df[7]
df.drop(columns=[7], inplace=True)
df.head()


# In[5]:


df.columns=features
df.head()


# In[6]:


df.drop(columns=["date"], inplace=True)
df.head()


# In[7]:


features_vectors=df.values


# In[8]:


X_train, X_test, Y_train, Y_test= train_test_split(features_vectors, targets ,test_size=0.1,random_state=1)


# In[9]:


reg=linear_model.LinearRegression()
reg.fit(X_train, Y_train)
print("Slope:", reg.coef_[0])
print("Intercept:", reg.intercept_)
print("Socre Training: ", reg.score(X_train, Y_train))
print("Socre Testing: ", reg.score(X_test, Y_test))


# In[10]:


F_values, p_values =f_regression(X_train ,Y_train)
print("F values:", F_values)
print("p values:", p_values)


# In[11]:


features_and_f_values= list(zip(df.columns,F_values))
features_and_f_values


# In[12]:


features_num_seq = range(1, len(features))
result_test_scores = list()
result_training_scores = list()
for num in features_num_seq:
    num_of_choosen_features = num
    selected_features = [
        feature_and_f_value[0]
        for feature_and_f_value in features_and_f_values[:num_of_choosen_features]
    ]

    features_vectors = df[selected_features].values
    X_train, X_test, Y_train, Y_test = train_test_split(features_vectors, targets, test_size=0.1, random_state=1)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)

    result_training_scores.append(reg.score(X_train, Y_train))
    result_test_scores.append(reg.score(X_test, Y_test))


# In[13]:


plt.plot(features_num_seq, result_training_scores, marker='o', label='train')
plt.plot(features_num_seq, result_test_scores, marker='*', label='test')

plt.xticks(features_num_seq)
plt.legend()
plt.xlabel('Number of features used')
plt.ylabel('Score')
plt.show()


# In[14]:


df_test=pd.read_csv("data/prediction.csv", delim_whitespace=True, header=None)
df_test.head()


# In[15]:


df_test.columns=features
df_test.head()
Date=df_test["date"]
Date


# In[16]:


df_test.drop(columns=["date"],inplace=True)
df_test.head()


# In[17]:


x=df_test.values
y_te_pred=reg.predict(x)
print(y_te_pred.shape)
print(x.shape)


# In[18]:


df_test


# In[19]:


prediction = pd.DataFrame(y_te_pred, columns=["peak_load(MW)"])
prediction


# In[20]:


result = pd.concat([ Date, prediction], axis=1)
result


# In[21]:


result.to_csv("submission.csv", index=False)


# In[ ]:
