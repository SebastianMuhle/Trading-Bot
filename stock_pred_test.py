import pandas as pd
from sklearn import preprocessing
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)

# Model trains and predicts based on the last 50 days of trading
history_points = 50


def csv_to_dataset(csv_path):
    print("Start csv_to_dataset")
    # Read the csv
    data = pd.read_csv(csv_path)
    print(data.head())
    print(data.shape)

    # Read S&P 500 and calculate the daily change rate
    stock_index = pd.read_csv('SP500.csv')
    open_price = stock_index['Open']
    change = (open_price - open_price.shift(1)) / open_price.shift(1) * 100
    stock_index['change_S&P500'] = change
    stock_index.dropna(inplace=True)
    del (stock_index['Open'])
    del (stock_index['High'])
    del (stock_index['Low'])
    del (stock_index['Close'])
    del (stock_index['Adj Close'])
    del (stock_index['Volume'])

    print(stock_index.head())

    # Merge
    #data = pd.merge(data, stock_index, on='timestamp')
    print("Before reordering")
    print(data.head())
    print(data.shape)
    print("After reordering")
    data = data.reindex(index=data.index[::-1])
    print(data.head())


    # Drop the timestamp of the data
    # It is not necessary for the model to know this
    data = data.drop('timestamp', axis=1)
    # Drop the first data point
    # It might contain an IPO which has a high volume that later effects the scaling of the volumen variable
    data = data.drop(0, axis=0)

    # TODO Necessary?
    data = data.values

    # Normalise between 0 and 1
    # This guarantees for nice gradients and thereby faster convergence of the model
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    # TODO replace 'ohlcv_histories_normalised' with 'training window'
    # TODO and the comment # Containing open high low close volume
    print('Length data normalised', len(data_normalised))
    ohlcv_histories_normalised = np.array(
        [data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array(
        [data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    print('Shape ohlcv', ohlcv_histories_normalised.shape)
    # Used for plotting later
    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    # Used to unnormalized the predicted values
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        technical_indicators.append(np.array([sma]))
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == \
           technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

# Get the data
ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')

# Train-Test Set split
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)
ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]
print("Training data")
print(y_train)
ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]
unscaled_y_test = unscaled_y[n:]
print("Test data")
print(unscaled_y_test)
print(ohlcv_train.shape)
print(ohlcv_test.shape)

# define two sets of inputs
# TODO change it from 5 to 6 to account for the new S&P 500 feature
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

# the first branch operates on the first input
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# the second branch opreates on the second input
y = Dense(20, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1, verbose=2)
# evaluation

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

# TODO Fix last plot

from datetime import datetime
model.save(f'technical_model.h5')
plt.figure()
complet_real = plt.plot(unscaled_y[start:n], label='real')
complet_pred = plt.plot(y_predicted[start:end], label='predicted')
