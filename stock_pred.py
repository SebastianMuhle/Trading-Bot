from data_preprocessing import csv_to_dataset
from model import build_model
import numpy as np
np.random.seed(4)


def train_bot(file_path, history_points, number_of_epochs, two_lstm_layers, number_of_neurons_lstm,
              two_layers_second_branch, number_of_neurons_second_branch, dropout_rate, s_and_p_500, ma7, ma21,
              ma_his_window, ema12, ema26, mac, ten_day_momentum,
              upper_bands, lower_bands):
    # Model trains and predicts based on the last 50 days of trading


    # Get the data
    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(file_path,
                                                                                                           history_points,
                                                                                                           s_and_p_500,
                                                                                                           ma7, ma21,
                                                                                                           ma_his_window,
                                                                                                           ema12, ema26,
                                                                                                           mac,
                                                                                                           ten_day_momentum,
                                                                                                           upper_bands,
                                                                                                           lower_bands
                                                                                                           )

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

    # Calls the function to build the model
    number_of_lstm_features = 5
    if s_and_p_500:
        number_of_lstm_features += 1

    model = build_model(history_points=history_points, technical_indicators=technical_indicators,
                        two_lstm_layers=two_lstm_layers, number_of_neurons_lstm=number_of_neurons_lstm,
                        two_layers_second_branch=two_layers_second_branch,
                        number_of_neurons_second_branch=number_of_neurons_second_branch, dropout_rate=dropout_rate)
    # Runs the training loop
    model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=number_of_epochs, shuffle=True, validation_split=0.1, verbose=2)

    # evaluation
    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    # Not necessary if you don't want to also print the training results
    #y_predicted = model.predict([ohlcv_histories, technical_indicators])
    #y_predicted = y_normaliser.inverse_transform(y_predicted)
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    print("Finished training")

    return scaled_mse, unscaled_y_test, y_test_predicted, ohlcv_test, tech_ind_test, model, y_normaliser
