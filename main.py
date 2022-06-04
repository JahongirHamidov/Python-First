import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datatime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# Load data
company = 'FB'

start = dt.datatime(2012, 1, 1)
end = dt.datatime(2020, 1, 1)

data = DataReader(company, 'Yahoo', start, end)

# Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'])

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)) :
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)