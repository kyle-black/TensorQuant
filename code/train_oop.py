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
        return bc.get_dollar_bars(self.time_bar_df, 10e+10)
    
    
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
        self.triple_result =barriers.apply_triple_barrier(self.bars_df,[1,1,1], 30)
        return self.triple_result
    
    def sample_weights(self):
        self.triple_result = self.triple_barriers()
        weights = return_attribution(self.triple_result)
        return weights
    


    

class Model:
    def __init__(self, bars_df):
        self.bars_df = bars_df
        #self.weights =weights

    def train_model(self):
        #output =adaboost_classifier(self.bars_df)
        #output = random_forest_classifier(self.bars_df)
        output =random_forest_classifier(self.bars_df)
     
        #output =random_forest_anomaly_detector(self.bars_df)
        return output

    




if __name__ == "__main__":
    stock = pd.read_csv('data/EURUSD.csv')
    stock.dropna(inplace=True)

    #print(stock.isna().any())
    
    
    
    bar_creator = CreateBars(stock)
    #x =bar_creator.dollar_bars()
   # print(x)
    
    
    time_bars_df = bar_creator.time_bars()

    
    #vol_bars_df = bar_creator.vol_bars()
    #dollar_bars_df = bar_creator.dollar_bars()


    #dollar_bars_df.to_csv('dol_bars.csv')

    
    
    
   

    
    analysis_instance_time = Analysis(time_bars_df)
    #print("Std Dev of Time Bars:", analysis_instance_time.std_dev())
    #print('Jaque Bera test:', analysis_instance_time.jaque_bera() )
    #print('ADF Statistic:', analysis_instance_time.AD_fuller() )

    print('ACF Statistic:', analysis_instance_time.acf() )


    feature_instance_time = FeatureMaker(time_bars_df, 30)
    print(feature_instance_time.feature_add())
    
    '''
    feature_bars =feature_instance_time.feature_add()
    feature_instance_time.elbow_()



    #d =feature_instance_time.fac_diff()
    #df =feature_bars.fac_diff()
   # print(feature_bars)
    #print(d)
   # print(feature_bars)

   # print(feature_bars)
    
    
   # print(feature_bars.columns)
    
    
    label_instance_time =Labeling(feature_bars)
    label_instance_time = label_instance_time.triple_barriers()
    
    print(label_instance_time)
    
    #df, weights = label_instance_time.sample_weights()
    #print(label_instance_time)
   # print(weights)

    model =Model(label_instance_time)
    
    print(model.train_model())
    '''

    

   










    

    

