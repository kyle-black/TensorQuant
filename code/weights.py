import pandas as pd
import numpy as np
from datetime import datetime as dt


def return_attribution(df):
    value_gains = []
    df  = df.copy()
    df.dropna(subset=['t1'], inplace =True)

    
    idx_list =[]
    no_ = []
    fd = df['t1'].tolist()

    dl = []
    for i in fd:

        try: dl.append(df.loc[i, 'Close'])

        except:
            no_.append(i)
            print(i)
            #print(i)
    #x = df.loc[fd,'Close']

  
    
    for idx, value in df.iterrows():
        #future_date = value['t1']

        
        future_date =pd.to_datetime(value['t1'])
        #df.[]
        
        
        #idx_list.append(idx)        
        
        #furure_
       # print(idx)
        #future_date = value['t1'].dt.date

        #future_date = value['t1']
        #print((future_date)) 
        #price_idx = future_date
        #cross_price = df.loc[price_idx, 'Close']

        #print(cross_price)    
        
        if value['label'] == 0:

            try:
                future_close = df.loc[future_date, 'Close']
                #value_gain = future_close - value['Close'] 
                value_gain =1
                value_gains.append(value_gain)
            except KeyError:
                future_close =np.nan
        
        else:
            price_idx = value['cross_above_ma']

            try:
                future_close = df.loc[future_date, 'Close']
                #value_gain = 0
            except KeyError:
                future_close =np.nan





            cross_price = df.loc[price_idx, 'Close']
            value_gain = cross_price - value['Close']


            value_gains.append(value_gain)

    df['value_gain'] = value_gains
    df['weighted_label'] = df['value_gain']

    #df['weighted_label'].to_dict()

    return df,df['weighted_label']


    