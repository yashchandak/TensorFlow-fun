# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:11:12 2016

@author: yash
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

sess = tf.InteractiveSession()


"""
Convolutional Neural Net 
"""

def weight_variable(shape):
    #return initialised weight variable
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    #retuens initialised bias variables
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    #returns result of convolving x with W
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
def max_pool_2x2(x):
    #returns pooled values from x
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding='SAME')

def validate(inp, out, accuracy, i = -1):
    #validating at ith step and logging progress
    train_accuracy = accuracy.eval(feed_dict={x:inp, y_:out, keep_prob:1.0})
    print("step %d, training accuracy %g" %(i, train_accuracy))
 

def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/'+name, mean)
    tf.scalar_summary('max/'+name, tf.reduce_max(var))
    tf.histogram_summary(name, var)
    

x  = tf.placeholder(tf.float32, shape=[None,784])   #Input
tf.image_summary('input', x, 10)
y_ = tf.placeholder(tf.float32, shape=[None,10])    #Expected outcome 
x_image = tf.reshape(x, [-1,28,28,1])               #reshape input vector


#First convolution layer with 32 filters of size 5x5
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) 

#second convloutional layer with 64 filters of size 5x5
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) 

#fully connected layer with 1024 hidden units
W_fc1 = weight_variable([7*7*64, 1024]) 
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])    #Flatten the result of convolution
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropouts
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop =  tf.nn.dropout(h_fc1, keep_prob)

#final classifier
W_fc2 = weight_variable([1024, 10]) 
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
sess.run(tf.initialize_all_variables())

for i in range(500):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        validate(batch[0], batch[1], accuracy, i)
    train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
    
validate(mnist.test.images, mnist.test.labels, accuracy)
                 