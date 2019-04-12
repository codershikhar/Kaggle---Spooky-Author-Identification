#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df


# In[4]:


X = df[['id', 'text']]
X


# In[5]:


y = df[['author']]


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=30000, ngram_range=(1, 3))
vectorizer


# In[7]:


X = vectorizer.fit_transform(df['text'])
X


# In[8]:


X = X.toarray()
X


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf


# In[11]:

print('fitting')
rf.fit(X_train, y_train)


# In[ ]:

print('fitting done')
print('Train Score - ', rf.score(X_train, y_train))
print('Test Score - ', rf.score(X_test, y_test))