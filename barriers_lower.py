import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay, DateOffset, WeekOfMonth, LastWeekOfMonth
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import Day, BMonthEnd

stock_market_holidays = [
    # 2003
    "2003-01-01", "2003-01-20", "2003-02-17", "2003-04-18", "2003-05-26", "2003-07-04", "2003-09-01", "2003-11-27", "2003-12-25",

    # 2004
    "2004-01-01", "2004-01-19", "2004-02-16", "2004-04-09", "2004-05-31", "2004-07-05", "2004-09-06", "2004-11-25", "2004-12-24",

    # 2005
    "2005-01-01", "2005-01-17", "2005-02-21", "2005-03-25", "2005-05-30", "2005-07-04", "2005-09-05", "2005-11-24", "2005-12-26",

    # 2006
    "2006-01-02", "2006-01-16", "2006-02-20", "2006-04-14", "2006-05-29", "2006-07-04", "2006-09-04", "2006-11-23", "2006-12-25",

    # 2007
    "2007-01-01", "2007-01-15", "2007-02-19", "2007-04-06", "2007-05-28", "2007-07-04", "2007-09-03", "2007-11-22", "2007-12-25",

    # 2008
    "2008-01-01", "2008-01-21", "2008-02-18", "2008-03-21", "2008-05-26", "2008-07-04", "2008-09-01", "2008-11-27", "2008-12-25",

    # 2009
    "2009-01-01", "2009-01-19", "2009-02-16", "2009-04-10", "2009-05-25", "2009-07-03", "2009-09-07", "2009-11-26", "2009-12-25",

    # 2010
    "2010-01-01", "2010-01-18", "2010-02-15", "2010-04-02", "2010-05-31", "2010-07-05", "2010-09-06", "2010-11-25", "2010-12-24",

    # 2011
    "2011-01-01", "2011-01-17", "2011-02-21", "2011-04-22", "2011-05-30", "2011-07-04", "2011-09-05", "2011-11-24", "2011-12-26",

    # 2012
    "2012-01-02", "2012-01-16", "2012-02-20", "2012-04-06", "2012-05-28", "2012-07-04", "2012-09-03", "2012-11-22", "2012-12-25",

    # 2013
    "2013-01-01", "2013-01-21", "2013-02-18", "2013-03-29", "2013-05-27", "2013-07-04", "2013-09-02", "2013-11-28", "2013-12-25",

    # 2014
    "2014-01-01", "2014-01-20", "2014-02-17", "2014-04-18", "2014-05-26", "2014-07-04", "2014-09-01", "2014-11-27", "2014-12-25",

    # 2015
    "2015-01-01", "2015-01-19", "2015-02-16", "2015-04-03", "2015-05-25", "2015-07-03", "2015-09-07", "2015-11-26", "2015-12-25",

    # 2016
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25", "2016-05-30", "2016-07-04", "2016-09-05", "2016-11-24", "2016-12-26",

    # 2017
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-04-14", "2017-05-29", "2017-07-04", "2017-09-04", "2017-11-23", "2017-12-25",

    # 2018
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-03-30", "2018-05-28", "2018-07-04", "2018-09-03", "2018-11-22", "2018-12-25",

    # 2019
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04", "2019-09-02", "2019-11-28", "2019-12-25",

    # 2020
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25", "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",

    # 2021
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31", "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",

    # 2022
    "2022-01-01", "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",

    # 2023
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
]

def apply_triple_barrier(df, pt_sl, num_days_active):
    """
    Apply the triple barrier method to label events.
    
    Parameters:
    - df: DataFrame with price data.
    - pt_sl: List of multipliers for profit taking and stop-loss.
    - num_days_active: Number of days the barrier should be kept active.
    
    Returns:
    - DataFrame with timestamps and their respective labels.
    """
   # return df

    # Ensure the index is a DatetimeIndex
    #df.set_index('Date', inplace=True)
    #return df['Date']
    df.set_index('Date', inplace=True)
    
    
    df.index = pd.to_datetime(df.index)

    price_series = df['Close'].copy()
   
    # Compute daily volatility
    daily_volatility = price_series.pct_change().ewm(span=num_days_active).std()

    # Initialize barriers DataFrame
    barriers = pd.DataFrame(index=df.index)

    for timestamp, price in price_series.items():
        volatility = daily_volatility.loc[timestamp]
        
        upper_barrier = price * (1 + pt_sl[0] * volatility)
        lower_barrier = price * (1 - pt_sl[1] * volatility)

        # Assign upper and lower barriers to the DataFrame
        barriers.at[timestamp,'price'] = price
        barriers.at[timestamp, 'volatility'] = volatility
        
        barriers.at[timestamp, 'upper_barrier'] = upper_barrier
        barriers.at[timestamp, 'lower_barrier'] = lower_barrier

       
        
        # Compute vertical barrier
        t1_date = timestamp
        for _ in range(num_days_active):
            t1_date += pd.Timedelta(days=1)
            while t1_date not in df.index:
                t1_date += pd.Timedelta(days=1)
                if t1_date > df.index[-1]:
                    t1_date = df.index[-1]
                    break
        
        barriers.at[timestamp, 't1'] = t1_date

        df_temp = df.loc[timestamp:t1_date]
        
        touch_upper = df_temp[df_temp['High'] >= upper_barrier].index.min()
        touch_lower = df_temp[df_temp['Low'] <= lower_barrier].index.min()

       # cross_above_ma = df_temp[(df_temp['Close'].shift(1) < df_temp['Middle_Band']) & (df_temp['Close'] > df_temp['Middle_Band'])].index.min()
       # cross_below_ma = df_temp[(df_temp['Close'].shift(1) > df_temp['Middle_Band']) & (df_temp['Close'] < df_temp['Middle_Band'])].index.min()

        barriers.at[timestamp, 'touch_upper'] = touch_upper
        barriers.at[timestamp, 'touch_lower'] = touch_lower


      #  barriers.at[timestamp, 'cross_above_ma'] = cross_above_ma
      #  barriers.at[timestamp, 'cross_below_ma'] = cross_below_ma

        if touch_lower != pd.NaT and (touch_upper == pd.NaT or touch_lower < touch_upper) and touch_lower < t1_date:
            barriers.at[timestamp, 'label'] = 1
        #if touch_upper != pd.NaT and (touch_lower == pd.NaT or touch_upper < touch_lower) and touch_upper < t1_date:
        #    barriers.at[timestamp, 'label'] = 1
        else:
            barriers.at[timestamp, 'label'] = 0
        
            

        
        
         # Assuming a default label if none of the conditions is met
    
    df_merged = df.join(barriers, how='left')
    return df_merged

