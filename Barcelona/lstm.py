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

from tensorflow.contrib import learn as tflearn

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# hyperparameters
lr = 0.0001
#batch_size = 513 # 2565 / 5
numBatchs = 5
#batchSizTest = len(XtestSet)
n_classes = 10
n_inputs = 10   # MNIST data input (img shape: 28*28)
n_steps = 20    # time steps
#n_hidden_units = 6   # neurons in hidden layer

# Define weights

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #, stddev=0.1 / w['in'] = tf.random_normal
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1) #, shape=shape
    return tf.Variable(initial)

# this is data
paddType = 1
nChannels = 1
reader = dt.ReaderTS(n_inputs)
XdataSet, YdataSet = reader.load_csvdata(n_steps)

print XdataSet['train'].shape, YdataSet['train'].shape
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
#lens = tf.placeholder(tf.int32, [None])
initBatch = tf.placeholder(tf.int32, shape=())

with open('ResultsLSTM.csv', 'wb') as csvfile:
    arrfieldnames = np.array(['training_iters', 'n_hidden_units', 'train error','testError'])
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(arrfieldnames)
    for training_iters in [1000]:
        for n_hidden_units in [2,5,10,20]: # seems like 5 and 20 working better
            varName = str(training_iters) + '_'+ str(n_hidden_units)
            with tf.variable_scope(str(varName)):
                # basic LSTM Cell.
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True, activation=tf.tanh)
                _init_state = lstm_cell.zero_state(initBatch, dtype=tf.float32)
                output, final_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=_init_state, time_major=False)

                # hidden layer for output as the final results
                W_out = weight_variable([n_hidden_units, n_classes])
                b_out = bias_variable([n_classes])
                outputs = tf.unstack(tf.transpose(tf.sigmoid(output),[1,0,2])) #tf.reshape(tf.sigmoid(output),[-1,n_steps])
                pred = tf.matmul(outputs[-1], W_out) + b_out

                ###  Optimizer  ###
                loss = tf.losses.mean_squared_error(y, pred) #target, predictions
                eval_metric_ops = {"rmse": tf.metrics.root_mean_squared_error(y, pred)}

                train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                            global_step=tf.contrib.framework.get_global_step(),
                                            learning_rate=lr, optimizer="SGD")
                '''
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                train_op = tf.train.AdamOptimizer(lr).minimize(cost)
                '''
                correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                
                #prediction, loss = tflearn.models.linear_regression(output, y)
                print "Before running"
                init = tf.initialize_all_variables()
                with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    step = 0
                    while step < training_iters:
                        '''
                        indexAccSet = np.reshape(batchDataSets[0:numBatchs-1],(batch_size*(numBatchs-1)))
                        batch_xs = XdataSet[indexAccSet]
                        batch_ys = YdataSet[indexAccSet]
                        '''
                        #sess.run([train_op], feed_dict={x: batch_xs,y: batch_ys,initBatch: (batch_size*(numBatchs-1))})
                        sess.run([train_op], feed_dict={x: XdataSet['train'],y: YdataSet['train'], initBatch:XdataSet['train'].shape[0]})
                        if step % (training_iters/10) == 0:
                            trnresult = sess.run( accuracy, feed_dict={x: XdataSet['val'], y: YdataSet['val'], initBatch:XdataSet['val'].shape[0]})
                            print(trnresult)
                        step += 1
                        '''
                    XtestSet = XdataSet[batchDataSets[numBatchs-1]]
                    YtestSet = YdataSet[batchDataSets[numBatchs-1]]
                    '''
                    tstresult =sess.run(accuracy, feed_dict={x: XdataSet['test'], y: YdataSet['test'], initBatch:XdataSet['test'].shape[0]})
                    print "Test accuracy: " + str(tstresult)
                    spamwriter.writerow([training_iters, n_hidden_units, round(trnresult,3), round(tstresult,3)])

