from PIL import Image
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

baseDir = os.path.abspath('.')
logreg = LogisticRegression(penalty='l1',solver='liblinear',C=0.62, max_iter = 50)

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
        img = img.resize((64,64), Image.BILINEAR)
        img = img.convert('L')
        img = np.resize(img,64*64)
        X.append(np.array(img)/255.)
        y.append(labels['gender'][int(item.split('.')[0])])
    y = np.array(y)
    Y = np.array((y+1)/2)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=0)
    return x_train, x_test, y_train, y_test

def train(x_train,y_train):
    logreg.fit(x_train, y_train)
    y_pred= logreg.predict(x_train)
    acc = accuracy_score(y_train,y_pred)
    print('Accuracy on training set: '+str(acc))
    return acc
    
    
def test(x_test,y_test):
    y_pred= logreg.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    print('Accuracy on test set: '+str(acc))
    return acc
