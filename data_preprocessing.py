import pandas as pd
from sklearn import preprocessing
import numpy as np
np.random.seed(4)


def csv_to_dataset(csv_path, history_points):
    print("Start csv_to_dataset")
    # Read the csv
    data = pd.read_csv(csv_path)
    print(data.head())
    print(data.shape)

    # Read S&P 500 and calculate the daily change rate
    stock_index = pd.read_csv('data/SP500.csv')
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

