from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers


def build_model(history_points, technical_indicators, number_of_lstm_features=5, two_lstm_layers=False,
                number_of_neurons_lstm=50, two_layers_second_branch=False, number_of_neurons_second_branch=20,
                dropout_rate=0.2):
    # Defining the two sets of inpus
    # TODO change it from 5 to 6 to account for the new S&P 500 feature
    lstm_input = Input(shape=(history_points, number_of_lstm_features), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # First branch operating on the first input
    x = LSTM(number_of_neurons_lstm, name='lstm_0', return_sequences=True)(lstm_input)
    x = Dropout(dropout_rate, name='lstm_dropout_0')(x)
    if two_lstm_layers:
        x2 = LSTM(number_of_neurons_lstm, name='lstm_1')(x)
        lstm_branch = Model(inputs=lstm_input, outputs=x2)
    else:
        lstm_branch = Model(inputs=lstm_input, outputs=x)

    # Second branch operating on the second input
    y = Dense(number_of_neurons_second_branch, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(dropout_rate, name='tech_dropout_0')(y)
    if two_layers_second_branch:
        y2 = Dense(number_of_neurons_second_branch, name='tech_dense_1')(y)
        y2 = Activation("relu", name='tech_relu_1')(y2)
        technical_indicators_branch = Model(inputs=dense_input, outputs=y2)
    else:
        technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)


    # Combining the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    # Defining two fully-connected layers
    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # Adding a layers together into a model
    # Defining an optimizer for the model
    # Compiling the model
    # Returning the model
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    return model
