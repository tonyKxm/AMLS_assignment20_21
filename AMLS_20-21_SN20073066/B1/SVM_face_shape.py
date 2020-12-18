import cv2
import joblib
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

#load pre-trained model
baseDir = os.path.abspath('.')
clfPath = os.path.join(baseDir ,'B1','faceShape.m')
clf = joblib.load(clfPath)

def preProcessing():
    X=[]
    y=[]
    filePath = os.path.join(baseDir ,'Datasets','cartoon_set','img')
    filelist = os.listdir(filePath)
    labelPath = os.path.join(baseDir ,'Datasets','cartoon_set','labels.csv') 
    labels = pd.read_csv(labelPath,delimiter = '\t')
    #kernel for dilation
    kernel = np.ones((3,3),np.uint8) 
    for item in filelist:
        path = os.path.join(filePath,item)
        img= cv2.imread(path)
        #resize and convert image to 64*64 grayscale image.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #remove noises and backgrounds
        _,mask = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.dilate(img,kernel,iterations = 3)
        img = cv2.resize(img, (64, 64)) 
        img = np.array(img)
        #flatten to 1D array
        img = img.reshape(64*64)
        X.append(img)
        #read labels according to image name
        label = int(labels['face_shape'][int(item.split('.')[0])])
        y.append(label)
    Y = np.array(y)
    x_train, x_vali, y_train, y_vali = train_test_split(X, Y,random_state=0)
    
    #following code reads test set
    X=[]
    y=[]
    filePath = os.path.join(baseDir ,'Datasets','cartoon_set_test','img')
    filelist = os.listdir(filePath)
    labelPath = os.path.join(baseDir ,'Datasets','cartoon_set_test','labels.csv') 
    labels = pd.read_csv(labelPath,delimiter = '\t')
    kernel = np.ones((3,3),np.uint8) 
    for item in filelist:
        path = os.path.join(filePath,item)
        img= cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.dilate(img,kernel,iterations = 3)
        img = cv2.resize(img, (64, 64)) 
        img = np.array(img)
        img = img.reshape(64*64)
        X.append(img)
        label = int(labels['face_shape'][int(item.split('.')[0])])
        y.append(label)
    Y = np.array(y)
    x_test = X
    y_test = Y
    #return all datasets
    return x_train, x_vali, x_test, y_train, y_vali, y_test

def train(x_train,y_train): 
    pred = clf.predict(x_train)
    #check acc in training set
    acc = accuracy_score(y_train, pred)
    return acc

def validation(x_vali,y_vali): 
    #check acc in validation set
    pred = clf.predict(x_vali)
    acc = accuracy_score(y_vali, pred)
    return acc

def test(x_test,y_test):
    #check acc in test set
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    return acc
