"""
# Youtube video tutorial: https://www.youtube.com/watch?v=Se9ByBnKb0o&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f

This code is a modified version of both tutorials:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT
"""
import csv
import dataSet_ts as dt
import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# hyperparameters
lr = 0.001
#batch_size = 513 # 2565 / 5
numBatchs = 5
#batchSizTest = len(XtestSet)
n_classes = 10
n_inputs = 10   # MNIST data input (img shape: 28*28)
n_steps = 20    # time steps
n_hidden_units = 6   # neurons in hidden layer

# Define weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #, stddev=0.1 / w['in'] = tf.random_normal
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1) #, shape=shape
    return tf.Variable(initial)

def my_model(features, target, params):
    print params
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True, activation=tf.tanh)
    _init_state = lstm_cell.zero_state(params['batch_size'], dtype=tf.float32)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, features, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    W_out = weight_variable([n_hidden_units, n_classes])
    b_out = bias_variable([n_classes])
    outputs = tf.unstack(tf.transpose(tf.sigmoid(output),[1,0,2])) #tf.reshape(tf.sigmoid(output),[-1,n_steps])
    pred = tf.matmul(outputs[-1], W_out) + b_out

    ###  Optimizer  ###
    ## I also could use this one I guess 
    #tf.contrib.learn.models.logistic_regression_zero_init(features, target)
    loss = tf.losses.mean_squared_error(target, pred) #target, predictions
    eval_metric_ops = {"rmse": tf.metrics.root_mean_squared_error(target, pred)}

    train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                global_step=tf.contrib.framework.get_global_step(),
                                learning_rate=lr, optimizer="SGD")
    '''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    '''
    correct_pred = tf.equal(pred,target)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #From the blog: http://terrytangyuan.github.io/2016/06/09/scikit-flow-v09/
    #(predictions, loss, train_op)
    return pred, loss, train_op

# this is data
paddType = 1
nChannels = 1
reader = dt.ReaderTS(n_inputs)
XdataSet, YdataSet = reader.load_csvdata(n_steps)

x_train, x_test, y_train, y_test = XdataSet['train'],XdataSet['test'],YdataSet['train'],YdataSet['test']

classifier = learn.Estimator(model_fn=my_model, params={'batch_size':x_train.shape[0]})
classifier.fit(x_train, y_train, steps=700)

predictions = classifier.predict(x_test)