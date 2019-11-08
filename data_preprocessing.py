import pandas as pd
from sklearn import preprocessing
import numpy as np
np.random.seed(4)

def calc_ema(values, time_period):
        sma = np.mean(values[:,3])
        #ma50 = np.mean(his[:, 3])

        ema_values = [sma]
        k = 2 / (1 + time_period)

        for i in range(len(values) - time_period, len(values)):
            close = values[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]


def csv_to_dataset(csv_path, history_points, s_and_p_500, ma7, ma21, ma_his_window, ema12, ema26, mac, ten_day_momentum,
                   upper_bands, lower_bands, volatilty_index_feature):
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
    # Drop unnecessary columns
    del (stock_index['Open'])
    del (stock_index['High'])
    del (stock_index['Low'])
    del (stock_index['Close'])
    del (stock_index['Adj Close'])
    del (stock_index['Volume'])

    print(stock_index.head())

    # Volatitly index
    volatilty_index = pd.read_csv('data/SP500.csv')
    del (volatilty_index['High'])
    del (volatilty_index['Low'])
    del (volatilty_index['Close'])
    del (volatilty_index['Adj Close'])
    del (volatilty_index['Volume'])

    volatilty_index = volatilty_index.rename(columns={"Date": "timestamp", "Open": "Vol_Index"})

    # Merge
    if s_and_p_500:
        data = pd.merge(data, stock_index, on='timestamp')
    if volatilty_index_feature:
        data = pd.merge(data, volatilty_index, on='timestamp')
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
    # TODO and add the comment # Containing open high low close volume
    print('Length data normalised', len(data_normalised))
    ohlcv_histories_normalised = np.array(
        [data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    ohlcv_histories_unnormalised = np.array(
        [data[i:i + history_points].copy() for i in range(len(data) - history_points)])
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
    # TODO add an input parameter, that leads you decide which technical indicators to choose from
    for his in ohlcv_histories_unnormalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        ma7 = np.mean(his[-7:, 3])
        ma21= np.mean(his[-21:, 3])
        ma_his_window = np.mean(his[:, 3])

        ema12 = calc_ema(his, 12)
        ema26 = calc_ema(his, 26)
        macd = calc_ema(his, 12) - calc_ema(his, 26)

        ten_day_momentum = his[-1, 3]/his[-10, 3]

        std20 = np.std(his[-20:, 3])
        upper_bands = ma21 + std20*2
        lower_bands = ma21 - std20*2

        technical_indicators.append(np.array([ma7, ma21, ma_his_window,ema12, ema26,macd,ten_day_momentum,upper_bands, lower_bands,  ]))

    technical_indicators = np.array(technical_indicators)

    # Drop not selected features
    delete = []
    if not ma7:
        delete.append(0)
    if not ma21:
        delete.append(1)
    if not ma_his_window:
        delete.append(2)
    if not ema12:
        delete.append(3)
    if not ema26:
        delete.append(4)
    if not mac:
        delete.append(5)
    if not ten_day_momentum:
        delete.append(6)
    if not upper_bands:
        delete.append(7)
    if not lower_bands:
        delete.append(8)

    technical_indicators_dropped = np.delete(technical_indicators, delete, 1)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators_dropped)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == \
           technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
