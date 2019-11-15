from numpy.random import seed
seed(1)
import random as rn
rn.seed(12345)
import tensorflow as tf
tf.random.set_seed(1)
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers


def build_model(history_points, technical_indicators, number_of_lstm_features=5, two_lstm_layers=False,
                number_of_neurons_lstm=50, two_layers_second_branch=False, number_of_neurons_second_branch=20,
                dropout_rate=0.2):
    '''
    This function builds the model using the Keras functional API.
    The detail working of the function is describe with comments in the code
    :param history_points: How long is the time frame you want to choose
    :param technical_indicators: The technical_indicators are used to determine the input shape of the normal
    neural network branch
    :param number_of_lstm_features: Used to determine the shape of the lstm_input.
    :param two_lstm_layers: If true two lstm_layers are used
    :param number_of_neurons_lstm: How many neurons are used in the LSTM layers
    :param two_layers_second_branch: If true two layers are used for the normal neural network branch
    :param number_of_neurons_second_branch: How many neurons are used in the neural network branch
    :param dropout_rate: Determines the dropout rate used in the dropout layers
    :return: Returns a compiled Keras model
    '''
    # Defining the two sets of inpus
    lstm_input = Input(shape=(history_points, number_of_lstm_features), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # First branch operating on the first input. This is the LSTM part
    if two_lstm_layers:
        print("Two layers")
        print(two_lstm_layers)
        x = LSTM(number_of_neurons_lstm, name='lstm_0', return_sequences=True)(lstm_input)
        x = Dropout(dropout_rate, name='lstm_dropout_0')(x)
        x2 = LSTM(number_of_neurons_lstm, name='lstm_1')(x)
        lstm_branch = Model(inputs=lstm_input, outputs=x2)
    else:
        x = LSTM(number_of_neurons_lstm, name='lstm_0')(lstm_input)
        x = Dropout(dropout_rate, name='lstm_dropout_0')(x)
        lstm_branch = Model(inputs=lstm_input, outputs=x)

    # Second branch operating on the second input. This is the normal Neural Network part
    y = Dense(number_of_neurons_second_branch, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(dropout_rate, name='tech_dropout_0')(y)
    if two_layers_second_branch:
        y2 = Dense(number_of_neurons_second_branch, name='tech_dense_1')(y)
        y2 = Activation("relu", name='tech_relu_1')(y2)
        technical_indicators_branch = Model(inputs=dense_input, outputs=y2)
    else:
        technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # Combining the output of the two branches to create the hybrid-network
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    # Defining two fully-connected layers on top of the two branches
    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # Define the final model to get a Keras model object
    # Defining an optimizer for the model
    # Compiling the model, using the optimizer and adding a loss to it
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    # Returning the model
    return model
