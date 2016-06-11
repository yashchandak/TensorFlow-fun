# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:53:27 2016

@author: yash
"""


"""
Fully connected Neural Net
"""


import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

sess = tf.InteractiveSession()

x  = tf.placeholder(tf.float32, shape=[None,784])   #Input
y_ = tf.placeholder(tf.float32, shape=[None,10])    #Expected outcome 

W  = tf.Variable(tf.zeros([784,10]))                #weights
b  = tf.Variable(tf.zeros([10]))                    #Biases

sess.run(tf.initialize_all_variables())             #start session, initialise variables

y  = tf.nn.softmax(tf.matmul(x,W) + b)              #Compute class probablities
cross_entropy  = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1])) #Error function

#Trainin =g the neural network
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(10):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})


#Evaluating performance    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))



            
