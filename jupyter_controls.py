from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def dropdown_widget():

    # Dummy function that is required
    def f(x):
        print(x)

    widget = interactive(f, x=widgets.Dropdown(
        options=['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DOW', 'XOM', 'GS',
                 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
                 'UTX', 'UNH', 'VZ'],
        description='Symbol'));
    return widget
