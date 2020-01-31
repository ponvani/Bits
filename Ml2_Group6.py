#!/usr/bin/env python
# coding: utf-8

# 
# ML Assignment #2
# 
# Group ID : ML_GROUP006
# Group Members:
# 
# 1. Ponvani : 2018AC04559 
# 2. Bala Kavin Pon : 2018AC04531
# 3. Poornima J : 2018AC04550
# 4. Venkataramanan Krishnan : 2018AC04529
# 
# Problem Statement: Predict whether the patient has diabetes or not.
# 
# Import the data from Indian diabetes dataset (Links to an external site.) and find dataset description from here (Links to an external site.) (2 points).
# 
# Consider all columns as independent variables and assign to variable X except the last column and consider the last column as the dependent variable and assign to variable y. Remove columns which don’t help the problem statement. (1 point).
# 
# Do Feature Scaling on Independent variables (2 points).
# 
# Split the data into train and test dataset (1 point).
# 
# Use Keras to make the neural network model and train the dataset on the same. (4 points).
# 
# Compute the accuracy and confusion matrix. (2 points).

# Import Libraries

# In[80]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.externals import joblib


# Step 1: Loading Data

# In[81]:


df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None);
print(df)


# Adding Header to the data

# In[82]:


df.columns = ["pregnancies_times", "glucose", "bp", "skin_thickness","insulin","bmi","Diabetes_Pedigree_Function","age","outcome"]
df.head()


# In[83]:


df.shape


# In[84]:


df.info()


# In[85]:


df.describe().T


#  Glucose,BP,skin_thickness,Insulin & bmi have min value as 0 which does not seem to be a valid value for these attributes so replace these values with atrribute mean values

# In[86]:


for cols in ['glucose','bp','skin_thickness','insulin','bmi']:
    df.loc[df[cols] == 0,cols]= df[cols].mean(skipna=True)
df.describe().T


# In[87]:


df.corr()


# In[88]:


corr_matrix = df.corr()
sns.heatmap(corr_matrix);


# Extract X and Y colummns

# In[89]:


from sklearn.model_selection import train_test_split
Y = df['outcome']
X = df.drop(columns=['outcome'])


# Test Train Split 

# In[90]:


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
print('X_train size {} , X_test size {}'.format(X_train.shape,X_test.shape))


# #Normalizing the data

# In[91]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)


# In[92]:


print (Y)


# In[93]:


#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))


# In[95]:


import keras
keras.__version__


# In[ ]:




