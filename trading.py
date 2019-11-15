import numpy as np
np.random.seed(4)


def trading(ohlcv_test, tech_ind_test, y_normaliser, model, purchase_amt, trading_cost):
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
    thresh = trading_cost + 0.1

    start = 0
    end = -1

    x = -1
    # Used to make sure that the bot does not perform to sell after enoughter, because the bot sells all stock in a
    # sale trade
    last_trade = "Empty"
    # Creates the traiding process and performs the trades
    for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]])
        price_today = y_normaliser.inverse_transform(normalised_price_today)
        predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))
        delta = predicted_price_tomorrow - price_today
        if delta > thresh:
            buys.append((x, price_today[0][0]))
            last_trade = "Buy"
        elif delta < -thresh and last_trade == "Buy":
            sells.append((x, price_today[0][0]))
            last_trade = "Sell"
        x += 1

    print('The numbers of buys and sells that our bot would have taken')
    print(f"buys: {len(buys)}")
    print(f"sells: {len(sells)}")

    # Compute the earnings
    def compute_earnings(buys_, sells_):
        stock = 0
        balance = 0
        while len(buys_) > 0 and len(sells_) > 0:
            if buys_[0][0] < sells_[0][0]:
                balance -= purchase_amt
                stock += purchase_amt / buys_[0][1]
                buys_.pop(0)
            else:
                balance += stock * sells_[0][1]
                stock = 0
                sells_.pop(0)
        # Sell the last stock if you still have something in your account
        # Makes sure, that all stock is sold and we can compare the returns
        # One buy sell order open, but no buy order left
        if len(sells_) > 0:
            balance += stock * sells_[0][1]
            stock = 0
            sells_.pop(0)
        total_trading_cost = (len(buys) + len(sells)) * trading_cost
        print('The money our bot would have earned')
        print(f"earnings: ${balance - total_trading_cost}")

    # we create new lists so we dont modify the original
    compute_earnings([b for b in buys], [s for s in sells])

    return buys, sells
