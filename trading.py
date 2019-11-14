import numpy as np
np.random.seed(4)


def trading(ohlcv_test, tech_ind_test, y_normaliser, model, purchase_amt, trading_cost):
    buys = []
    sells = []
    thresh = trading_cost + 0.1

    start = 0
    end = -1

    x = -1
    last_trade = "Empty"
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

    def compute_earnings(buys_, sells_):
        stock = 0
        balance = 0
        while len(buys_) > 0 and len(sells_) > 0:
            #print("Buy:" + str(buys_[0][0]) + " Sell:" + str(sells_[0][0]))
            if buys_[0][0] < sells_[0][0]:
                # time to buy $10 worth of stock
                balance -= purchase_amt
                stock += purchase_amt / buys_[0][1]
                #print("Buy Balance:" + str(balance) + " Stock:" + str(stock))
                buys_.pop(0)
            else:
                # time to sell all of our stock
                balance += stock * sells_[0][1]
                stock = 0
                #print("Sell Balance:" + str(balance) + " Stock:" + str(stock))
                sells_.pop(0)
        # Sell the last stock if you still have something in your account
        # Makes sure, that all stock is sold and we can compare the returns
        # One buy sell order open, but no buy order left
        if len(sells_) > 0:
            balance += stock * sells_[0][1]
            stock = 0
            #print("Sell Balance:" + str(balance) + " Stock:" + str(stock))
            sells_.pop(0)
        total_trading_cost = (len(buys) + len(sells)) * trading_cost
        print('The money our bot would have earned')
        print(f"earnings: ${balance - total_trading_cost}")

    # we create new lists so we dont modify the original
    compute_earnings([b for b in buys], [s for s in sells])

    return buys, sells
