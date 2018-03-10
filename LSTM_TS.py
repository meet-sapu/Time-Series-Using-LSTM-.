# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 01:46:00 2018

@author: 5558
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import math



data = pd.read_csv("F:/time_series/Train_SU63ISt.csv",header=0)
test_data = pd.read_csv('F:/time_series/Test_0qrQsBZ.csv')
data = data.loc[:,'Datetime':'Count']
scaler = MinMaxScaler(feature_range=(0, 1))
temp_count = scaler.fit_transform(np.reshape(data['Count'],(18288,1)))
data['Count'] = temp_count
data['Datetime'] = pd.to_datetime(data['Datetime'], infer_datetime_format=True)

data = data.set_index('Datetime')

ts = pd.Series(data['Count'])
ts = ts.astype('float64')

#Plotting Decomposition .
decomposition = seasonal_decompose(ts,freq = 365*24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


def create_dataset(data,look_back):
    datax,datay = [],[]
    for i in range(len(data)-look_back-1):
         a=data[i:(i+look_back)].values
         datax.append(a)
         datay.append(data[i + look_back])
    return np.array(datax), np.array(datay)

look_back = 24*30
trainx , trainy = create_dataset(ts,look_back)

trainx = np.reshape(trainx, (trainx.shape[0],1, trainx.shape[1]))

#Building LSTM Network .
model = Sequential()
model.add(LSTM(6 , input_shape =(1,look_back) , return_sequences = True))
model.add(LSTM(3,return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainx, trainy, epochs=50, batch_size=4, verbose=2)


trainPredict = model.predict(trainx)

trainPredict = scaler.inverse_transform(trainPredict)
trainy = scaler.inverse_transform([trainy])
trainy = np.transpose(trainy)

trainScore = math.sqrt(mean_squared_error(trainy, trainPredict))


#Foreasting
final_predictions = []
i =0
while(i < len(test_data)):
    if i == 0 :
        gg = temp_count[len(temp_count)-look_back:len(temp_count),0]
        tgg = np.reshape(gg,(1,1,look_back))
        res = model.predict(tgg)
        final_predictions.append(np.reshape(res,(1))[0])
        i = i + 1
    else:
        temp = np.append(gg[1:look_back] , res)           
        gg = temp
        tgg = np.reshape(gg,(1,1,look_back))
        res = model.predict(tgg)
        final_predictions.append(np.reshape(res,(1))[0])
        i = i + 1

final_prediction = pd.Series(final_predictions)
final_prediction = np.reshape(final_prediction,(5112,1))
final_prediction.astype('float64')
final_prediction = scaler.inverse_transform(final_prediction)
final = test_data
final['Predictions'] = final_prediction

writer = pd.ExcelWriter('F:/time_series_LSTM.xlsx')
final.to_excel(writer)




