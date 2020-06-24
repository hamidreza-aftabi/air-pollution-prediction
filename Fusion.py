from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
'''from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error'''
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,merge
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU,Flatten
import numpy as np
import matplotlib.pyplot as plt
import time
import keras

data = np.load('polution_dataSet.npy')

'Layer3------------------------------------------------------------------------------'
X3 = np.zeros((40000, 56))
y3 = np.zeros((40000, 1))
for i in range(35000):
    for j in range(7):
        X3[i, 8 * j:8 * j + 8] = data[i + 1008 - (6 - j) * 168, :]
    y3[i,] = data[i + 1176, 0]

n_hours = 7
n_features = 8
n_train_hours = 10000

train_X3 = X3[:n_train_hours, :]
valid_X3 = X3[10000:12000, :]
test_X3 = X3[12000:14000, :]

train_y3 = y3[:n_train_hours, :]
valid_y3 = y3[10000:12000, :]
test_y3 = y3[12000:14000, :]

train_X3 = train_X3.reshape((train_X3.shape[0], n_hours, n_features))
test_X3= test_X3.reshape((test_X3.shape[0], n_hours, n_features))
valid_X3 = valid_X3.reshape((valid_X3.shape[0], n_hours, n_features))

'Layer 2------------------------------------------------------------------------------'

X2 = np.zeros((40000, 56))
y2 = np.zeros((40000, 1))
for i in range(35000):
    for j in range(7):
        X2[i, 8 * j:8 * j + 8] = data[i + 144 - (6 - j) * 24, :]
    y2[i,] = data[i + 168, 0]

n_hours = 7
n_features = 8
n_train_hours = 10000
train_X2 = X2[:n_train_hours, :]
valid_X2 = X2[10000:12000, :]
test_X2 = X2[12000:14000, :]

train_y2 = y2[:n_train_hours, :]
valid_y2 = y2[10000:12000, :]
test_y2 = y2[12000:14000, :]

train_X2 = train_X2.reshape((train_X2.shape[0], n_hours, n_features))
test_X2= test_X2.reshape((test_X2.shape[0], n_hours, n_features))
valid_X2 = valid_X2.reshape((valid_X2.shape[0], n_hours, n_features))

'Layer 1------------------------------------------------------------------------------'
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
values1 = reframed.values
n_train_hours=10000
train1 = values1[:n_train_hours, :]
valid1=values1[n_train_hours:12000, :]
test1= values1[12000:14000, :]
n_obs = n_hours * n_features

train_X1, train_y1 = train1[:, :n_obs], train1[:, -n_features]
valid_X1, valid_y1 = valid1[:, :n_obs], valid1[:, -n_features]
test_X1, test_y1 = test1[:, :n_obs], test1[:, -n_features]

train_X1 = train_X1.reshape((train_X1.shape[0], n_hours, n_features))
valid_X1 = valid_X1.reshape((valid_X1.shape[0], n_hours, n_features))
test_X1 = test_X1.reshape((test_X1.shape[0], n_hours, n_features))

'--------------------------------------------------------------------------------------------------------------'
inp1 = Input(shape=(train_X1.shape[1], train_X1.shape[2]))
inp2 = Input(shape=(train_X2.shape[1], train_X2.shape[2]))
inp3 = Input(shape=(train_X3.shape[1], train_X3.shape[2]))

gr1=CuDNNGRU(256,return_sequences=True)(inp1)
gr2=CuDNNGRU(256,return_sequences=True)(inp2)
gr3=CuDNNGRU(256,return_sequences=True)(inp3)


out1=Dense(1,kernel_initializer='normal',activation='sigmoid')(gr1)
out2=Dense(1,kernel_initializer='normal',activation='sigmoid')(gr2)
out3=Dense(1,kernel_initializer='normal',activation='sigmoid')(gr3)

mrg = keras.layers.concatenate([out1,out2,out3],axis=1)

mrg=Flatten()(mrg)

out=Dense(1,kernel_initializer='normal',activation='sigmoid')(mrg)

model = Model(input=[inp1, inp2, inp3], output=out)

model.compile(optimizer='Adam',
              loss='mae')

history = model.fit([train_X1,train_X2,train_X3], train_y2, epochs=50,validation_data=([valid_X1,valid_X2,valid_X3],valid_y2), batch_size=72, verbose=True, shuffle=False)

from keras.utils import plot_model
plot_model(model, to_file='Fusion.png')

yhat = model.predict([test_X1,test_X2,test_X3])
plt.figure()
x=np.arange(test_y2.shape[0])
plt.plot(x,test_y2,label='True')
plt.plot(x,yhat,label='Predicted')
plt.title('True and Predicted Value for Test Data of Fusioned GRU Network')
plt.xlabel('Times')
plt.ylabel('Values')
plt.legend()

# plot history
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss of  Fusioned GRU Network')
plt.title('Train and Validation Loss of  Fusioned GRU Network')
plt.legend()


plt.show()
