import cv2
import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#load pre-trained model
baseDir = os.path.abspath('.')
modelPath = os.path.join(baseDir ,'B2','model')
metaPath = os.path.join(modelPath ,'eye-model-9999.meta')

def preProcessing():
    X=[]
    y=[]
    filePath = os.path.join(baseDir ,'Datasets','cartoon_set','img')
    filelist = os.listdir(filePath)
    labelPath = os.path.join(baseDir ,'Datasets','cartoon_set','labels.csv') 
    labels = pd.read_csv(labelPath,delimiter = '\t')
    for item in filelist:
        path = os.path.join(filePath,item)
        img= cv2.imread(path)
        #extract eye area and resize into 20*20
        eye = img[220:300,150:250]
        img = cv2.resize(eye, (20, 20))
        #data normalization
        X.append(img/255.)
        zero = np.zeros([5])
        #one-hot encoding
        zero[int(labels['eye_color'][int(item.split('.')[0])])] = 1
        y.append(zero)
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
        eye = img[220:300,150:250]
        img = cv2.resize(eye, (20, 20))
        X.append(img/255.)
        zero = np.zeros([5])
        zero[int(labels['eye_color'][int(item.split('.')[0])])] = 1
        y.append(zero)
    Y = np.array(y)
    x_test = X
    y_test = Y
    return x_train, x_vali, x_test, y_train, y_vali, y_test

def train(x_train,y_train):
    #check acc in training set
    #clear gpu memory
    tf.reset_default_graph()
    with  tf.Session() as sess:
        saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, tf.train.latest_checkpoint(modelPath))

        graph = tf.get_default_graph()
        input_images = graph.get_tensor_by_name('input_images:0')
        result = graph.get_tensor_by_name('result:0')
        
        feed_dict = {input_images:x_train, result:y_train}

        acc = graph.get_operation_by_name('acc').outputs[0]
        acc = sess.run(acc, feed_dict)
        
        return acc

def validation(x_vali,y_vali):
    #check acc in validation set
    with  tf.Session() as sess:
        saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, tf.train.latest_checkpoint(modelPath))

        graph = tf.get_default_graph()
        input_images = graph.get_tensor_by_name('input_images:0')
        result = graph.get_tensor_by_name('result:0')
        feed_dict = {input_images:x_vali, result:y_vali}

        acc = graph.get_operation_by_name('acc').outputs[0]
        acc = sess.run(acc, feed_dict)
        
        return acc
    
def test(x_test,y_test):
    #check acc in test set
    with  tf.Session() as sess:
        saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, tf.train.latest_checkpoint(modelPath))

        graph = tf.get_default_graph()
        input_images = graph.get_tensor_by_name('input_images:0')
        result = graph.get_tensor_by_name('result:0')
        feed_dict = {input_images:x_test, result:y_test}

        acc = graph.get_operation_by_name('acc').outputs[0]
        acc = sess.run(acc, feed_dict)
        
        return acc
