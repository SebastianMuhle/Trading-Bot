from data_preprocessing import csv_to_dataset
from model import build_model
import numpy as np
np.random.seed(4)

# Model trains and predicts based on the last 50 days of trading
history_points = 50

# Get the data
ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('data/daily_MSFT.csv', history_points)

# Train-Test Set split
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)
ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]
ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]
unscaled_y_test = unscaled_y[n:]

model = build_model(history_points, technical_indicators)
model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=2, shuffle=True, validation_split=0.1, verbose=2)

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

# TODO WRITE THE OUTPUT AND BUYING DECSIONS IN A CSV FILE

from datetime import datetime
#model.save(f'technical_model.h5')
#plt.figure()
#all_predict = np.append(y_predicted[start:end], y_test_predicted[start:end], axis=0)
#print(all_predict.shape)
#complet_real = plt.plot(unscaled_y[start:n], label='real')
#complet_pred = plt.plot(y_predicted[start:end], label='predicted')
#complet_pred = plt.plot(all_predict, label='predicted')

#plt.show()
