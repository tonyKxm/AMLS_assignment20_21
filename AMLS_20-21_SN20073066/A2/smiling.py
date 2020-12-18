from PIL import Image
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

#load pre-trained model
baseDir = os.path.abspath('.')
modelPath = os.path.join(baseDir ,'A2','model')
metaPath = os.path.join(modelPath ,'smiling-model-5900.meta')

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
        #array normalization
        X.append(np.array(img)/255.)
        #read labels according to image name
        y.append(labels['smiling'][int(item.split('.')[0])])
    y = np.array(y)
    #male = 1 and female =0
    Y = np.array([(y+1)/2, (1-y)/2]).T
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
        X.append(np.array(img)/255.)
        y.append(labels['smiling'][int(item.split('.')[0])])
    y = np.array(y)
    Y = np.array([(y+1)/2, (1-y)/2]).T
    x_test = X
    y_test = Y
    #return all datasets
    return x_train, x_vali, x_test, y_train, y_vali, y_test


def train(x_train,y_train):
    #check acc in training set
    #clear gpu memory
    tf.reset_default_graph()
    with  tf.Session() as sess:
        #restore pre-trained model
        saver = tf.train.import_meta_graph(metaPath)
        saver.restore(sess, tf.train.latest_checkpoint(modelPath))

        graph = tf.get_default_graph()
        #get tensors
        input_images = graph.get_tensor_by_name('input_images:0')
        result = graph.get_tensor_by_name('result:0')
        
        #feed variables into model
        feed_dict = {input_images:x_train, result:y_train}

        #get acc
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
