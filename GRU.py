from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
'''from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
import numpy as np
import matplotlib.pyplot as plt
import time

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

#-------------------------------------------------------------------------------------------------------------------
# design network
model = Sequential()
model.add(CuDNNGRU(256, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
t1=time.clock()
history = model.fit(train_X, train_y, epochs=60, batch_size=72, validation_data=(valid_X, valid_y), verbose=True, shuffle=False)
t2=time.clock()
# plot history
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss of GRU Network')
plt.legend()

print('Training Time For GRU:',(t2-t1))

yhat = model.predict(test_X)
plt.figure()
plt.scatter(test_y,yhat)
plt.title('True VS Predicted Value for Test Data of GRU Network')
plt.xlabel('True value')
plt.ylabel('Predicted value')

plt.figure()
x=np.arange(test_y.shape[0])
plt.plot(x,test_y,label='True')
plt.plot(x,yhat,label='Predicted')
plt.title('True and Predicted Value for Test Data of GRU Network')
plt.xlabel('Times')
plt.ylabel('Values')
plt.legend()


plt.show()

