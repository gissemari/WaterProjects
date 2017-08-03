#from tensorflow.contrib.learn.python.learn.datasets import base
#from tensorflow.python.framework import dtypes
import feature as ft
import numpy as np
import pandas as pd

class ReaderTS(object):

	def __init__(self, numDim):
		self.numDim = numDim

	@staticmethod
	def rnn_data( data, idxStart, idxEnd, time_steps, labels=False):
	    """
	    creates new data frame based on previous observation
	      * example:
	        l = [1, 2, 3, 4, 5]
	        time_steps = 2
	        -> labels == False [[1, 2], [2, 3], [3, 4]]
	        -> labels == True [3, 4, 5]
	    """
	    rnn_df = []
	    for i in range(idxStart, idxEnd): # not a-b because we take all the examples
	        if labels:
	            try:
	                rnn_df.append(data.iloc[i + time_steps].as_matrix())
	            except AttributeError:
	                rnn_df.append(data.iloc[i + time_steps])
	        else:
	            data_ = data.iloc[i: i + time_steps].as_matrix()
	            aux_len = len(data_.shape)
	            if aux_len > 1:
	            	aux = data_
	            	#print aux_len  
	            else:
	            	aux = [[i] for i in data_]
	            	print aux
	            rnn_df.append(aux)
	    return np.array(rnn_df, dtype=np.float32)

	@staticmethod
	def split_data(data, time_steps, val_size=0.1, test_size=0.1):
	    """
	    splits data to training, validation and testing parts
	    """
	    completeInstances = len(data) - time_steps
	    ntest = int(round( completeInstances* (1 - test_size)))
	    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

	    #df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
	    #print df_train.shape, df_val.shape, df_test.shape
	    #return df_train, df_val, df_test
	    return nval, ntest, completeInstances

	@staticmethod
	def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
	    """
	    Given the number of `time_steps` and some data,
	    prepares training, validation and test data for an lstm cell.
	    """
	    #df_train, df_val, df_test = ReaderTS.split_data(data, val_size, test_size)
	    split1, split2, split3 = ReaderTS.split_data(data, time_steps, val_size, test_size)
	    return (ReaderTS.rnn_data(data, 0, split1, time_steps, labels=labels),
	            ReaderTS.rnn_data(data, split1, split2, time_steps, labels=labels),
				ReaderTS.rnn_data(data, split2, split3, time_steps, labels=labels))

	def load_csvdata(self,time_steps, seperate=False):
	    f = open("10DMAs.csv")#rawdata
	    l = []
	    l = [ line.split(",") for line in f]
	    data=np.asarray(l).astype(float) ## rows: 10 - spots columns: 2880 points in time
	    data = np.transpose(data) # rows: time - columns: 10 spots
	    if not isinstance(data, pd.DataFrame):
	        data = pd.DataFrame(data)
	    print data.shape
	    train_x, val_x, test_x = ReaderTS.prepare_data(data, time_steps) 
	    #data['a'] if seperate else 
	    train_y, val_y, test_y = ReaderTS.prepare_data(data, time_steps, labels=True) 
	    #data['b'] if seperate else 
	    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)		
