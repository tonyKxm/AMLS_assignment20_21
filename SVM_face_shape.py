#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2


# In[2]:


def get_data():
    X=[]
    y=[]
    filelist = os.listdir('./cartoon_set/img')
    labels = pd.read_csv('./cartoon_set/labels.csv',delimiter = '\t')
    kernel = np.ones((3,3),np.uint8) 
    for item in filelist:
        path = os.path.join('./cartoon_set/img',item)
        img= cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.dilate(img,kernel)
        img = cv2.resize(img, (64, 64)) 
        img = np.array(img)
        img = img.reshape(64*64)
        X.append(img)
        label = int(labels['face_shape'][int(item.split('.')[0])])
        y.append(label)
    Y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=0)

    return x_train, x_test, y_train, y_test


# In[3]:


# x_train, x_test, y_train, y_test = get_data()
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
x_train, x_test, y_train, y_test = get_data()


# In[20]:


classifier = svm.SVC(kernel='rbf')
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred))


# In[7]:


import joblib
joblib.dump(classifier, "faceShape.m")


# In[12]:


clf = joblib.load("faceShape.m")
pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred))


# In[24]:


def test_all():
    X=[]
    y=[]
    filelist = os.listdir('./cartoon_set/img')
    labels = pd.read_csv('./cartoon_set/labels.csv',delimiter = '\t')
    kernel = np.ones((3,3),np.uint8) 
    for item in filelist:
        path = os.path.join('./cartoon_set/img',item)
        img= cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.dilate(img,kernel)
        img = cv2.resize(img, (64, 64)) 
        img = np.array(img)
        img = img.reshape(64*64)
        X.append(img)
        label = int(labels['face_shape'][int(item.split('.')[0])])
        y.append(label)
    Y = np.array(y)

    return X, Y


# In[25]:


X,Y = test_all()
Y_pred = classifier.predict(X)
print("Accuracy:", accuracy_score(Y, Y_pred))

