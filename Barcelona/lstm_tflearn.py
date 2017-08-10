from __future__ import division, print_function, absolute_import

import dataSet_ts as dt
import tflearn
import matplotlib.pyplot as plt
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# hyperparameters
lr = 0.001
#batch_size = 513 # 2565 / 5
numBatchs = 5
#batchSizTest = len(XtestSet)
n_classes = 10
n_inputs = 10   # MNIST data input (img shape: 28*28)
n_steps = 20    # time steps
n_hidden_units = 6   # neurons in hidden layer

# Barcelona Dataset loading
reader = dt.ReaderTS(n_inputs)
XdataSet, YdataSet = reader.load_csvdata(n_steps)

x_train, x_test, y_train, y_test = XdataSet['train'],XdataSet['test'],YdataSet['train'],YdataSet['test']
x_val, y_val = XdataSet['val'], YdataSet['val']


# Network building
net = tflearn.input_data([None,n_inputs ,n_steps])
net = tflearn.lstm(net, n_hidden_units) #, dropout=0.8
net = tflearn.fully_connected(net,n_classes, activation='linear')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='mean_square')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(x_train, y_train,  n_epoch=100, validation_set=(x_val, y_val), show_metric=True,
          batch_size=32)
predictions = model.predict(x_test) 
print(predictions)
print(predictions.shape)#(1179, 10)


fig, ax = plt.subplots(1)
fig.autofmt_xdate()

#Selecting just the first
plot_predicted, = ax.plot(predictions[0:100,0], label='prediction')

plot_test, = ax.plot(y_test[0:100,0], label='Real value')

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('Barcelona water prediction - 1st DMA')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
