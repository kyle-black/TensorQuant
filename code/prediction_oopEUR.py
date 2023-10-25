import pandas as pd
import numpy as np
import bar_creation as bc
from statistics import variance
import numpy as np
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import probplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
import barriers
import features
from train_models import random_forest_classifier #, adaboost_classifier, random_forest_ts #, random_forest_anomaly_detector
from weights import return_attribution

from autocorrelation import compute_and_plot_acf
import elbow_plot
import prediction_fit
import get_data
from datetime import datetime

import live_data

from redis_connect import redis_connection
import time
import schedule

class CreateBars:
    def __init__(self, raw_bars):
        self.raw_bars = raw_bars
        self.time_bar_df = None  # Initialize this attribute
        self.time_bar_dict = None
    def time_bars(self):
        self.time_bar_df = bc.time_bars(self.raw_bars)  # Set the attribute
        self.time_bar_dict = self.time_bar_df.to_dict('records')
        return self.time_bar_df

    def vol_bars(self):
        # Check if time_bar_df has been created, if not, create it
        if self.time_bar_df is None:
            self.time_bars()
        return bc.get_volume_bars(self.time_bar_df, 10)
    

    def dollar_bars(self):
        # Check if time_bar_df has been created, if not, create it
        if self.time_bar_df is None:
            self.time_bars()
        return bc.get_dollar_bars_P(self.time_bar_df, 2000)
    
    
class Analysis:
    def __init__(self, bars_df):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df

    def std_dev(self):
        # Ensure bars_df is a DataFrame and has a 'Close' column
        if isinstance(self.bars_df, pd.DataFrame) and 'Daily_Returns' in self.bars_df.columns:
            std_dev_value = np.std(self.bars_df['Daily_Returns'][1:])
            return std_dev_value
        else:
            raise ValueError("Provided data is not a valid DataFrame or doesn't have a 'Close' column.")
        
    def jaque_bera(self):   # Test for normality
        jb_stat, p_value, _, _ = jarque_bera(self.bars_df['Daily_Returns'][1:])
        return jb_stat, p_value, _, _ 
    def AD_fuller(self): # Check for Stationary
        result = adfuller(self.bars_df['Daily_Returns'][1:])
        return  result
    def acf(self):
        acf_val = compute_and_plot_acf(self.bars_df['Close'][1:])

        return acf_val

class FeatureMaker:
    def __init__(self, bars_df, window):
        # Store the bars dataframe regardless of its type (time, volume, dollar)
        self.bars_df = bars_df
        self.window =window

    def feature_add(self):

        results =features.add_price_features(self.bars_df, self.window)

        return results

    def fac_diff(self):
        f_results = self.feature_add()
       # df =features.fractional_diff(f_results)
        d_values = features.find_min_d_for_df(f_results)
        return d_values
    
    def elbow_(self):
        result = self.feature_add()
        return elbow_plot.plot_pca(result)

    




class Labeling:
    def __init__(self, bars_df):
        self.bars_df = bars_df
        

    def triple_barriers(self):
        self.triple_result =barriers.apply_triple_barrier_P(self.bars_df,[1,1,1], 90)
        return self.triple_result
    
    def sample_weights(self):
        self.triple_result = self.triple_barriers()
        weights = return_attribution(self.triple_result)
        return weights
    


    

class Model:
    def __init__(self, symbol,bars_df):
        self.bars_df = bars_df

        self.symbol = symbol
        # predict on last 10 entriesail()
        self.bars_df = self.bars_df.tail(10)

        #self.weights =weights


    def predict_values(self):
        self.predictions, self.probas =prediction_fit.make_predictions_up(self.symbol,self.bars_df)
       # self.predictions_dwn, self.probas_dwn =prediction_fit.make_predictions_dwn(self.symbol,self.bars_df)
        return self.predictions, self.probas, self.bars_df['upper_barrier'], self.bars_df['lower_barrier'], self.bars_df['Close']

    




#if __name__ == "__main__":


def run_predictions():
    symbol = 'EURUSD'
    #stock = get_data.get_json(symbol)

    stock = live_data.latest_data()
    #stock = pd.read_csv('data/SPY_new.csv')
    
    #stock['Date'] = stock['Date'].dt.strftime('%Y-%m-%d')
    #print(type(stock['Date'][0]))
    stock = stock.iloc[::]
    bar_creator = CreateBars(stock)
    
    
    
    dollar_bars_df = bar_creator.dollar_bars()
   
    
    feature_instance_time = FeatureMaker(dollar_bars_df, 30)
    
    #print(dollar_bars_df)
    
    feature_bars =feature_instance_time.feature_add()
    feature_instance_time.elbow_()

   

    
    feature_bars =feature_bars[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
       'Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
       'Log_Returns', 'SpreadOC', 'SpreadLH', 'MACD', 'Signal_Line_MACD',
       'RSI']]
    
    label_instance_time =Labeling(feature_bars)
    label_instance_time = label_instance_time.triple_barriers()
    

    print(label_instance_time)
    
    model =Model(symbol,label_instance_time)
    print(model.predict_values())

    output= model.predict_values()
    


    ################################# Parse Model output
    hard_predictions = output[0]
    last_hard_prediction = hard_predictions[-1]
    prob_predictions = output[1]
    dwn_prob = prob_predictions[-1][0]
    up_prob = prob_predictions[-1][2]
    neutral_prob = prob_predictions[-1][1]
    
    upper_barriers = output[2]
    lower_barriers = output[3]
    closes = output[4]

    last_lower_barrier = round(lower_barriers.iloc[-1], 4)
    last_upper_barrier = round(upper_barriers.iloc[-1], 4)
    last_close = closes.iloc[-1]
    date = closes.index[-1]
    date = date.strftime('%H:%M:%S')

    print(f'last close price:{last_close}\n last_upper_barrier: {last_upper_barrier} \n last_lower_barrier: {last_lower_barrier} \n predict_up: {up_prob} \n predict_down:{dwn_prob} \n neutral_prob:{neutral_prob} \n hard_prediction:{last_hard_prediction}' )

    output_dict = {}

    output_dict[date] = {'close': last_close, 'up_prob':up_prob,'dwn_prob':dwn_prob, 'neutral_prob':neutral_prob, 'upper_barrier':last_upper_barrier, 'lower_barrier':last_lower_barrier, 'hard_prediction':last_hard_prediction}

    print(output_dict)

    if redis_connection.exists(date):
        print(f' {date} is an already existing bar')
    else:
        print(f'New Bar Created at {date}')

    

    
    
    
    res1 = redis_connection.hset(
    date,
    mapping={
        'close': last_close,
        'up_prob': up_prob,
        'dwn_prob': dwn_prob,
        'neutral_prob': neutral_prob,
        'upper_barrier': last_upper_barrier,
        'lower_barrier': last_lower_barrier,
        'hard_prediction': last_hard_prediction,
        'time': date
    },
)
    


    res4 = redis_connection.hgetall(date)
    print('redis:',res4)


schedule.every(5).minutes.do(run_predictions)

while True:
    schedule.run_pending()
    time.sleep(1)




