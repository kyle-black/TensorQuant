from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import datetime




def latest_data_60():
    username = 'kpblack87'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)


    nifty_index_data = tv.get_hist(symbol='EURUSD',exchange='OANDA',interval=Interval.in_1_hour,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)


   # df.set_index('Date', inplace=True)
    return df
#print(df)


def latest_data_5():
    username = 'kpblack87'
    password = 'Biobio9034!'

    tv = TvDatafeed(username, password)


    nifty_index_data = tv.get_hist(symbol='EURUSD',exchange='OANDA',interval=Interval.in_5_minute,n_bars=10000)

    df= nifty_index_data

    df =df.reset_index()
    df.rename(columns = {'datetime':'Date','open': 'Open', 'high':'High', 'low':'Low', 'close':'Close','volume':'Volume'}, inplace=True)

    return df