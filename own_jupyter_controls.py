from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def dropdown_widget():
    '''
    This function returns an interactive Jupyter Notebook dropdown widget with all the Dow Jones
    Stock symbols to choose from.
    We choose a dropdown menu to make it easier for the user and to guarantee correct user input.
    '''

    # Dummy function that is required for the widget
    def f(x):
        print(x)
        return x

    widget = interactive(f, x=widgets.Dropdown(
        options=['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DOW', 'XOM', 'GS',
                 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
                 'UTX', 'UNH', 'VZ'],
        description='Symbol'))
    return widget


def generate_file_path_for_stock_data(ticker_symbol):
    '''
    This function generates and returns the file path for the stock data of the selected stock.
    '''
    file_path = 'data/daily_' + str(ticker_symbol) + '.csv'
    return file_path


def network_architecture_widget():
    '''
    This function returns an interactive Jupyter Notebook dropdown widget with all the Dow Jones
    Stock symbols to choose from.
    We choose a dropdown menu to make it easier for the user and to guarantee correct user input.
    '''

    # Dummy function that is required for the widget
    def f(x):
        print(x)
        return x

    widget = interactive(f, x=widgets.IntSlider(min=10, max=128, step=1, value=50),
                         description='Numbers of Neurons in the LSTM-layer')
    return widget