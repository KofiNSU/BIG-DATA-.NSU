#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import re
import xml.etree.cElementTree as et


# In[2]:


def parser(path):
    d = dict()
    root = et.parse(path)
    rows = root.findall('.//column')
    for row in rows:
        d.setdefault(list(row.attrib.values())[0],[]).append(row.text)
    df_xml = pd.DataFrame.from_dict(d)
    return df_xml
       
path_trainX = ['tkk_train_2016.xml','bank_train_2016.xml']
path_testY  = ['tkk_test_etalon.xml','banks_test_etalon.xml']

data_trainX = pd.DataFrame()
data_testY = pd.DataFrame()

for path in path_trainX:
    df = parser(path)
    data_trainX = pd.concat([df, data_trainX], ignore_index=True)
for path in path_testY:
    df = parser(path)
    data_testY = pd.concat([df,data_testY], ignore_index=True)


# In[3]:


# we process the the text dataset

def preprocess_text(text):
    
    text = []
    for sen in range(0,len(text)):
        # we remove all the special characters
        text = re.sub(r/'w,' '',str(text[sen]))
        #remove the single characters 
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        #remove single characters at a start
        
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
        # we substitute the multiple  space in each character
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        
        text = re.sub(r'^b\s+', '', text)
        text  = text.lower()
        
        text = text.split()
        
        text = [stemmer.lemmatize(word) for word in text]
        
        text = ' '.join(text)
        
        text.append(text)
        
        return text

 


# In[4]:



data_trainX['prune_text'] = data_trainX.text.apply(lambda row:(preprocess_text))

data_testY['prune_text'] = data_testY.text.apply(lambda row:(preprocess_text))


# In[5]:


#we  drop twitid also
data_trainX.drop(['twitid'],axis =1, inplace =True,errors = 'ignore')


# In[6]:


#we drop the id also

data_trainX.drop(['id'],axis =1,inplace = True, errors ='ignore')


# In[7]:


#we drop the date since it does give any informaytion
data_trainX.drop(['date'],axis=1,inplace=True,errors='ignore')


# In[8]:


# we add an end-targ to each row in the dataset for easy iddentification of row and to give more info for each row.
def end_targ(all_row):
    
    if (all_row =='1').any():
        
        end_targ = 1
       
    elif (all_row =='0').any():
        
        end_targ = 0
        
    else:
        
        end_targ = -1
        
            
        return(end_targ)
        


# In[9]:


# We add the end_targ to the test data
data_testY['end_targ'] = data_testY.apply(end_targ, axis=1)


# In[10]:


# we apply the end _targ to the train data
data_trainX['end_targ'] = data_trainX.apply(end_targ,axis =1)


# In[12]:


# we drop the all NANS
new_data_trainX = data_trainX.fillna(0)

print(new_data_trainX)


# In[13]:


new_data_testY = data_testY.fillna(0)
print(new_data_testY)


# In[14]:


import gensim
from sklearn.feature_extraction.text import CountVectorizer


# In[15]:


# we use the count vectorizer function and transform the test and train data into values after it has been proces

CV  = CountVectorizer()

x_trainX  = CV.fit_transform(new_data_trainX.text)

y_testY =   CV.fit_transform(new_data_testY.text)



 
    


# In[16]:


#fimd the shape of the test data.

y_testY.shape


# In[17]:


x_trainX.shape


# In[18]:


print(x_trainX[0])


# In[19]:


# using sklearn model of logistic regression for classification
from sklearn.linear_model import LogisticRegression


# In[20]:


model_lr  = LogisticRegression()


# In[21]:


# we fit the model on the test data
model_lr.fit(y_testY,new_data_testY.end_targ)


# In[22]:


#find the score of regression on the text data
model_lr.score(y_testY, new_data_testY.end_targ)


# In[ ]:




