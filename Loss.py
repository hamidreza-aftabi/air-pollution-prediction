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

Epoch=60


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
n_train_hours=7000
train = values[:n_train_hours, :]
valid=values[n_train_hours:8000, :]
test = values[8000:10000, :]
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
valid_X, valid_y = valid[:, :n_obs], valid[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
valid_X = valid_X.reshape((valid_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#---------------------------------------------------------------------------------------------------------------------------
# design network
model3 = Sequential()
model3.add(CuDNNLSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model3.add(Dense(1))

model3.compile(loss='mae', optimizer='Adam')

# fit network

history3 = model3.fit(train_X, train_y, epochs=Epoch, batch_size=72, validation_data=(test_X, test_y), verbose=True, shuffle=False)

'----------------------------'

# design network
model4 = Sequential()
model4.add(CuDNNLSTM(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model4.add(Dense(1))

model4.compile(loss='mse', optimizer='Adam')

# fit network

history4 = model4.fit(train_X, train_y, epochs=Epoch, batch_size=72, validation_data=(test_X, test_y), verbose=True, shuffle=False)

'----------------------------'

# plot history
plt.figure()
plt.plot(history3.history['loss'], label='MAE Train Loss')
plt.plot(history4.history['loss'], label='MSE Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss of LSTM with Different Loss Function')
plt.legend()

plt.figure()
plt.plot(history3.history['val_loss'], label='MAE Test Loss')
plt.plot(history4.history['val_loss'], label='MSE Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss of LSTM with Different Loss Function')
plt.legend()


yhat3 = model3.predict(test_X)
plt.figure()
plt.scatter(test_y,yhat3)
plt.title('True VS Predicted Value for Test Data of LSTM Network with MAE Loss Function')
plt.xlabel('True value')
plt.ylabel('Predicted value')

yhat4 = model4.predict(test_X)
plt.figure()
plt.scatter(test_y,yhat4)
plt.title('True VS Predicted Value for Test Data of LSTM Network with MSE Loss Function')
plt.xlabel('True value')
plt.ylabel('Predicted value')

plt.show()
