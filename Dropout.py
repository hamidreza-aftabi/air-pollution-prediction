from math import sqrt
from numpy import concatenate
from pandas import read_csv
import tensorflow
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
from keras.layers import Dropout
from keras.layers import SimpleRNN
import numpy as np
import matplotlib.pyplot as plt
import time

Epoch=50


data=np.load('polution_dataSet.npy')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg


n_hours = 24
n_features = 8
reframed = series_to_supervised(data, n_hours, 1)
values = reframed.values
n_train_hours=20000
train = values[:n_train_hours, :]
test = values[20000:30000, :]
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

#Droput------------------------------------------------------------------------------------------------------------------
# design network
model1 = Sequential()
model1.add(CuDNNLSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(Dropout(.3))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam')

# fit network
t1=time.clock()
history1 = model1.fit(train_X, train_y, epochs=Epoch, batch_size=72, validation_data=(test_X, test_y), verbose=True, shuffle=False)
t11=time.clock()


# plot history
plt.figure()
plt.plot(history1.history['loss'], label='Train')
plt.plot(history1.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss of LSTM Network with Dropout')
plt.legend()

yhat = model1.predict(test_X)
plt.figure()
plt.scatter(test_y,yhat)
plt.title('True VS Predicted Value for Test Data of LSTM Network with Dropout')
plt.xlabel('True value')
plt.ylabel('Predicted value')

plt.figure()
x=np.arange(test_y.shape[0])
plt.plot(x,test_y,label='True')
plt.plot(x,yhat,label='Predicted')
plt.title('True and Predicted Value for Test Data of LSTM Network with Dropout')
plt.xlabel('Times')
plt.ylabel('Values')
plt.legend()


print('Training Time For LSTM with Dropout:',(t11-t1))
plt.show()
