import numpy as np
np.random.seed(4)


def trading(ohlcv_test, tech_ind_test, y_normaliser, model, purchase_amt, leverage):
    '''
    This function controlls the traiding. First the trade are executed then the earning are calculated
    :param ohlcv_test: Data used for prediction
    :param tech_ind_test: Data used for prediction
    :param y_normaliser: Used to denormlize the data
    :param model: The trained model used for prediction
    :param purchase_amt: The buying amount of the bot for one trade
    :param trading_cost: The trading cost the bot faces
    :return: Buy and sell patterns for plotting
    '''
    # Initalize the trading arrays
    buys = []
    sells = []
    #thresh = trading_cost + 0.1

    start = 0
    end = -1

    x = -1
    # Used to make sure that the bot does not perform to sell after enoughter, because the bot sells all stock in a
    # sale trade
    last_trade = "Empty"
    count_right_long = 0
    count_false_long = 0
    count_right_short = 0
    count_false_short = 0
    total_earnings = []
    # Creates the traiding process and performs the trades
    for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]])
        price_today = y_normaliser.inverse_transform(normalised_price_today)
        predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))
        delta = predicted_price_tomorrow - price_today
        thresh = price_today * 0.0003
        normalised_price_tomorrow = ohlcv[0][0]
        normalised_price_tomorrow = np.array([[normalised_price_tomorrow]])
        price_tomorrow = y_normaliser.inverse_transform(normalised_price_tomorrow)
        if delta > thresh:
            earnings = price_tomorrow - price_today - thresh
            if earnings > 0:
                count_right_long += 1
            else:
                count_false_long += 1
            buys.append((x, price_today[0][0], predicted_price_tomorrow, price_tomorrow))
            total_earnings.append(earnings)
            last_trade = "Long"
        elif delta < -thresh:
            earnings = price_today - price_tomorrow - thresh
            earnings = earnings * leverage
            if earnings > 0:
                count_right_short += 1
            else:
                count_false_short += 1
            sells.append((x, price_today[0][0], predicted_price_tomorrow, price_tomorrow))
            total_earnings.append(earnings)
            last_trade = "Short"
        x += 1

    print('The numbers of buys and sells that our bot would have taken')
    print(f"buys: {len(buys)}")
    print(f"sells: {len(sells)}")

    # Compute the earnings
    def compute_earnings(buys_, sells_):
        total_earnings_output = sum(total_earnings)
        print('The money our bot would have earned')
        print(f"earnings: ${total_earnings_output}")
        print("Count_right_long:", count_right_long)
        print("Count_false_long:", count_false_long)
        print("Count_right_short:", count_right_short)
        print("Count_false_short:", count_false_short)



    # we create new lists so we dont modify the original
    compute_earnings([b for b in buys], [s for s in sells])

    return buys, sells
