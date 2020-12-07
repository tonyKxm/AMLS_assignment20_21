#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random


# In[2]:


def get_data():
    X=[]
    y=[]
    filelist = os.listdir('./celeba/img')
    labels = pd.read_csv('./celeba/labels.csv',delimiter = '\t')
    for item in filelist:
        path = os.path.join('./celeba/img',item)
        img=Image.open(path)  
        img = img.resize((64,64), Image.BILINEAR)
        img = img.convert('L')
        X.append(np.array(img)/255.)
        y.append(labels['smiling'][int(item.split('.')[0])])
    y = np.array(y)
    Y = np.array([(y+1)/2, (1-y)/2]).T
    x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=0)

    return x_train, x_test, y_train, y_test


# In[3]:


def allocate_weights_and_biases():

    kernal_1 = 16
    kernal_2 = 32
    kernal_3 = 64
    hidden_1 = 512 
    hidden_2 = 128 
    
    X = tf.placeholder("float", [None, 64, 64],name = 'input_images')
    Y = tf.placeholder("float", [None, 2],name = 'result')  
    x_input = tf.reshape(X,[-1,64,64,1])
    stddev = 0.01
    
    weights = {
        'conv_layer1':tf.Variable(tf.random_normal([3,3,1,kernal_1], stddev=stddev)),
        'conv_layer2':tf.Variable(tf.random_normal([3,3,kernal_1,kernal_2], stddev=stddev)),
        'conv_layer3':tf.Variable(tf.random_normal([3,3,kernal_2,kernal_3], stddev=stddev)),
        'hidden_layer1': tf.Variable(tf.random_normal([64 * 64, hidden_1], stddev=stddev)),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_1, hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([hidden_2, 2], stddev=stddev))
    }

    biases = {
        'bias_ConvLayer1':tf.Variable(tf.random_normal([kernal_1], stddev=stddev)),
        'bias_ConvLayer2':tf.Variable(tf.random_normal([kernal_2], stddev=stddev)),
        'bias_ConvLayer3':tf.Variable(tf.random_normal([kernal_3], stddev=stddev)),
        'bias_layer1': tf.Variable(tf.random_normal([hidden_1], stddev=stddev)),
        'bias_layer2': tf.Variable(tf.random_normal([hidden_2], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2], stddev=stddev))
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
    
    layer_3 = tf.add(tf.nn.conv2d(layer_2, weights['conv_layer3'],strides=[1,1,1,1],padding='SAME'), biases['bias_ConvLayer3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.max_pool(layer_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    flat =  tf.contrib.layers.flatten(layer_3)  
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


learning_rate = 1e-4
training_epochs = 30000
display_accuracy_step = 10
save_step = 100


# In[8]:


logits, X, Y = multilayer_perceptron()
variables   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables ]) * 1e-5
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)+lossL2,name = 'loss')
predict_label = tf.argmax(logits, 1,name = 'predict_label')
true_prediction = tf.equal(predict_label, tf.argmax(Y, 1),name = 'check')
acc = tf.reduce_mean(tf.cast(true_prediction, "float"),name = 'acc')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init_op = tf.global_variables_initializer()


# In[9]:


saver = tf.train.Saver(max_to_keep=4)
with tf.Session() as sess:
    sess.run(init_op)
    
    for epoch in range(training_epochs):
        idx=random.randint(0,3718)
        batch= random.randint(32,64)
        train_input = x_train[idx:(idx+batch)]
        train_labels = y_train[idx:(idx+batch)]
        
        accuracy, _ , cost = sess.run([acc, train_op, loss_op], feed_dict={X: train_input,Y: train_labels})
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(cost))
        
        if epoch % display_accuracy_step == 0:
            print("Accuracy: ",accuracy)
            
        if (epoch % save_step == 0 and epoch!=0):
            saver.save(sess, "model/gender_model", global_step=epoch)   
            
    print("Optimization Finished!")
    print("Test Accuracy:", acc.eval({X: x_test,Y: y_test}))


# In[11]:


with  tf.Session() as sess:
        saver = tf.train.import_meta_graph('model1/my-model-37800.meta')
        saver.restore(sess, tf.train.latest_checkpoint("model1/"))

        graph = tf.get_default_graph()
        input_images = graph.get_tensor_by_name('input_images:0')
        result = graph.get_tensor_by_name('result:0')
        feed_dict = {input_images:x_test, result:y_test}

        acc = graph.get_operation_by_name('acc').outputs[0]
        loss = graph.get_operation_by_name('loss').outputs[0]
        predict_label = graph.get_operation_by_name('predict_label').outputs[0]
        check =  graph.get_operation_by_name('check').outputs[0]
        acc,loss,predict_label,check = sess.run([acc,loss,predict_label,check], feed_dict)
        
        print("the accuracy is:",acc)
        print("the loss is:",loss)
        print("the predict_label is:",predict_label)
        print("the check result is:",check)


# In[ ]:




