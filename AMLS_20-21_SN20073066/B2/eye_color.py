#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
        eye = img[220:300,150:250]
        img = cv2.resize(eye, (20, 20))
        X.append(img/255.)
        zero = np.zeros([5])
        zero[int(labels['eye_color'][int(item.split('.')[0])])] = 1
        y.append(zero)
    Y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=0)

    return x_train, x_test, y_train, y_test


# In[3]:


def allocate_weights_and_biases():

    kernal_1 = 16
    kernal_2 = 32
    hidden_1 = 512  
    hidden_2 = 128 
    
    X = tf.placeholder("float", [None, 20, 20, 3],name = 'input_images')
    Y = tf.placeholder("float", [None, 5],name = 'result')  
    x_input = tf.reshape(X,[-1,20,20,3])
    stddev = 0.01
    
    weights = {
        'conv_layer1':tf.Variable(tf.random_normal([3,3,3,kernal_1], stddev=stddev)),
        'conv_layer2':tf.Variable(tf.random_normal([3,3,kernal_1,kernal_2], stddev=stddev)),
        'hidden_layer1': tf.Variable(tf.random_normal([800, hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, 5], stddev=stddev))
    }

    biases = {
        'bias_ConvLayer1':tf.Variable(tf.random_normal([kernal_1], stddev=stddev)),
        'bias_ConvLayer2':tf.Variable(tf.random_normal([kernal_2], stddev=stddev)),
        'bias_layer1': tf.Variable(tf.random_normal([hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([5], stddev=stddev))
    }
    
    return weights, biases, X, Y,x_input
    


# In[4]:


def multilayer_perceptron():
        
    weights, biases, X, Y, x_input = allocate_weights_and_biases()

    layer_1 = tf.add(tf.nn.conv2d(x_input, weights['conv_layer1'],strides=[1,1,1,1],padding='SAME'), biases['bias_ConvLayer1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.max_pool(layer_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    layer_2 = tf.add(tf.nn.conv2d(layer_1, weights['conv_layer2'],strides=[1,1,1,1],padding='SAME'), biases['bias_ConvLayer2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.max_pool(layer_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    flat =  tf.contrib.layers.flatten(layer_2)  
    full_layer1 = tf.add(tf.matmul(flat,weights['hidden_layer1']),biases['bias_layer1'])
    full_layer1 = tf.nn.relu(full_layer1)
    full_layer1 = tf.nn.dropout(full_layer1,0.5)
    
    full_layer2 = tf.add(tf.matmul(full_layer1,weights['hidden_layer2']),biases['bias_layer2'])
    full_layer2 = tf.nn.relu(full_layer2)
    full_layer2 = tf.nn.dropout(full_layer2,0.5)
    
    out_layer = tf.matmul(full_layer2, weights['out']) + biases['out']
    out_layer = tf.nn.softmax(out_layer)

    return out_layer, X, Y


# In[5]:


x_train, x_test, y_train, y_test = get_data()


# In[7]:


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-3,  global_step, decay_steps=10, decay_rate=0.3)
training_epochs = 2000
display_accuracy_step = 10
logits, X, Y = multilayer_perceptron()
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init_op = tf.global_variables_initializer()


# In[ ]:


import random
training_epochs = 20000
save_frequency = 500
with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=4)
    sess.run(init_op)
    for epoch in range(training_epochs):
        idx=random.randint(0,7468)
        batch= random.randint(16,32)
        train_input = x_train[idx:(idx+batch)]
        train_labels = y_train[idx:(idx+batch)]
        _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input,Y: train_labels})
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))
                
        if epoch % display_accuracy_step == 0:
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy: {:.3f}".format(accuracy.eval({X: x_train, Y: y_train})))
        if epoch % save_frequency == 0:
            saver.save(sess, "models/face-model", global_step=epoch)
    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test Accuracy:", accuracy.eval({X: x_test,Y: y_test}))