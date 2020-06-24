from math import sqrt
from numpy import concatenate
from pandas import read_csv
import tensorflow
from pandas import DataFrame
from pandas import concat
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.callbacks import TensorBoard
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

#RNN------------------------------------------------------------------------------------------------------------------
# design network
model2 = Sequential()
model2.add(SimpleRNN(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model2.add(Dense(1))
model2.compile(loss='mae', optimizer='adam')


t2=time.clock()
history2 = model2.fit(train_X, train_y, epochs=Epoch, batch_size=72, validation_data=(valid_X, valid_y), verbose=True, shuffle=False)
t22=time.clock()

# plot history
plt.figure()
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss of RNN Network')
plt.legend()

yhat2 = model2.predict(test_X)
plt.figure()
plt.scatter(test_y,yhat2)
plt.title('True VS Predicted Value for Test Data of RNN Network')
plt.xlabel('True value')
plt.ylabel('Predicted value')


plt.figure()
x=np.arange(test_y.shape[0])
plt.plot(x,test_y,label='True')
plt.plot(x,yhat2,label='Predicted')
plt.title('True and Predicted Value for Test Data of RNN Network')
plt.xlabel('Times')
plt.ylabel('Values')
plt.legend()


print('Training Time For RNN:',(t22-t2))
plt.show()
