from PIL import Image
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# read current work path
baseDir = os.path.abspath('.')
# load model and set parameters
logreg = LogisticRegression(penalty='l1',solver='liblinear',C=0.88, max_iter = 50)

def preProcessing():
    X=[]
    y=[]
    filePath = os.path.join(baseDir ,'Datasets','celeba','img')
    filelist = os.listdir(filePath)
    labelPath = os.path.join(baseDir ,'Datasets','celeba','labels.csv') 
    labels = pd.read_csv(labelPath,delimiter = '\t')
    for item in filelist:
        path = os.path.join(filePath,item)
        img=Image.open(path) 
        #resize image to 64*64, conver to grayscale
        img = img.resize((64,64), Image.BILINEAR)
        img = img.convert('L')
        #flatten to 1D array and normalization
        img = np.resize(img,64*64)
        X.append(np.array(img)/255.)
        #read labels according to image name
        y.append(labels['gender'][int(item.split('.')[0])])
    y = np.array(y)
    #male = 1 and female =0
    Y = np.array((y+1)/2)
    x_train, x_vali, y_train, y_vali = train_test_split(X, Y,random_state=0)
    
    #following code reads test set
    X=[]
    y=[]
    filePath = os.path.join(baseDir ,'Datasets','celeba_test','img')
    filelist = os.listdir(filePath)
    labelPath = os.path.join(baseDir ,'Datasets','celeba_test','labels.csv') 
    labels = pd.read_csv(labelPath,delimiter = '\t')
    for item in filelist:
        path = os.path.join(filePath,item)
        img=Image.open(path)  
        img = img.resize((64,64), Image.BILINEAR)
        img = img.convert('L')
        img = np.resize(img,64*64)
        X.append(np.array(img)/255.)
        y.append(labels['gender'][int(item.split('.')[0])])
    y = np.array(y)
    Y = np.array((y+1)/2)
    
    x_test = X
    y_test = Y
    #return all datasets
    return x_train, x_vali, x_test, y_train, y_vali, y_test


def train(x_train,y_train):
    #train model
    logreg.fit(x_train, y_train)
    #check acc in training set
    y_pred= logreg.predict(x_train)
    acc = accuracy_score(y_train,y_pred)   
    return acc

def validation(x_vali,y_vali):
    #check acc in validation set
    y_pred= logreg.predict(x_vali)
    acc = accuracy_score(y_vali,y_pred)   
    return acc  
    
def test(x_test,y_test):
    #check acc in test set
    y_pred= logreg.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    return acc
