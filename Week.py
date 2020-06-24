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
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import CuDNNGRU
import numpy as np
import matplotlib.pyplot as plt
import time

data = np.load('polution_dataSet.npy')

X = np.zeros((35000, 56))
y = np.zeros((35000, 1))
for i in range(35000):
    for j in range(7):
        X[i, 8 * j:8 * j + 8] = data[i + 1008 - (6 - j) * 168, :]
    y[i,] = data[i + 1176, 0]

n_hours = 7
n_features = 8
n_train_hours =10000
train_X = X[0:n_train_hours, :]
valid_X = X[10000:12000, :]
test_X = X[12000:14000, :]

train_y = y[:n_train_hours, :]
valid_y = y[10000:12000, :]
test_y = y[12000:14000, :]

print(train_X.shape, len(train_X), train_y.shape)
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
valid_X = valid_X.reshape((valid_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#-------------------------------------------------------------------------------------------------------------------
# design network
model = Sequential()
model.add(CuDNNLSTM(50,input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#model.add(Dropout(.3))
model.add(CuDNNLSTM(50))
#model.add(Dropout(.3))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
t1=time.clock()
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(valid_X, valid_y), verbose=True, shuffle=False)
t2=time.clock()
# plot history
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss of LSTM Network(Weekly Time Series)')
plt.legend()

print('Training Time For LSTM:',(t2-t1))

yhat = model.predict(test_X)
plt.figure()
plt.scatter(test_y,yhat)
plt.title('True VS Predicted Value for Test Data of LSTM Network(Weekly Time Series)')
plt.xlabel('True value')
plt.ylabel('Predicted value')


plt.figure()
x=np.arange(test_y.shape[0])
plt.plot(x,test_y,label='True')
plt.plot(x,yhat,label='Predicted')
plt.title('True and Predicted Value for Test Data of LSTM Network(Weekly Time Series)')
plt.xlabel('Times')
plt.ylabel('Values')
plt.legend()



plt.show()
