import pandas as pd
import numpy as np  

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff


def add_price_features(df, window_length):
    df =df.copy()
    ### Add autocorrelation / serial correlation
    autocorr_lag_10 = df['Close'].autocorr(lag=window_length)

    ############# Bolinger band Calc

    # Compute Bollinger Bands
    #window_length = 3  # Typically 20 for daily data
    #num_std_dev = 2    # Typically 2

    # Middle Band = n-day simple moving average (SMA)
    num_std_dev = 2

    df['Middle_Band'] = df['Close'].rolling(window=window_length).mean()

    # Upper Band = Middle Band + (standard deviation of price x 2)
    df['Upper_Band'] = df['Middle_Band'] + df['Close'].rolling(window=window_length).std() * num_std_dev

    # Lower Band = Middle Band - (standard deviation of price x 2)
    df['Lower_Band'] = df['Middle_Band'] - df['Close'].rolling(window=window_length).std() * num_std_dev

    #Log Returns
    df['Log_Returns'] = np.log(df['Close']/ df['Close'].shift(window_length))
    df['SpreadOC'] = df['Open'] / df['Close']
    df['SpreadLH'] = df['Low'] / df['High']


    #######################MACD 
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()



    #########################RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))


    #df['SMI'] = df['Close'] - ((df['High'] -df['Low']).rolling(window_length=window_length).median()) 


    lookback_period = window_length
    smoothing_period = window_length

    # Calculate the raw SMI
    high_low_diff = df['High'] - df['Low']
    close_minus_lowest_low = df['Close'] - high_low_diff.rolling(window=lookback_period).min()
    highest_high_minus_lowest_low = high_low_diff.rolling(window=lookback_period).max() - high_low_diff.rolling(window=lookback_period).min()

    raw_smi = (close_minus_lowest_low / highest_high_minus_lowest_low) * 100
    double_smooth_smi1 = raw_smi.ewm(span=smoothing_period, adjust=False).mean()
    double_smooth_smi2 = double_smooth_smi1.ewm(span=smoothing_period, adjust=False).mean()

    df['SMI'] = double_smooth_smi2


    ##### get weekday from dataframe

     # Convert 'Date' column to datetime if it's not already
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # [Your existing code...]

    # Extract weekday and hour
    if 'Date' in df.columns:
        df['weekday'] = df['Date'].dt.weekday + 1  # +1 to make it 1-7 instead of 0-6
        df['hour'] = df['Date'].dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        df['weekday'] = df.index.weekday + 1
        df['hour'] = df.index.hour
    else:
        raise ValueError("No datetime information found in DataFrame")








    return df


def get_weights(d, size):
    # Returns the weights for fractional differencing
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

'''
def fractional_diff(dataframe, differencing_value=0.1, threshold=1e-5):
    """
    Returns a DataFrame where each column has been fractionally differenced.
    """
    diffed_data = {}

    for col in dataframe.columns:
        print(col)

        if col != 'Date':
            series = dataframe[col]

            # Determine the weights
            weights = get_weights(differencing_value, series.shape[0])

            # Ensure weights are above the threshold
            weights = weights[np.abs(weights) > threshold].flatten()

            print(weights)

            # Fractionally difference using the computed weights
            diff_series = []
            for i in range(len(weights), series.shape[0]+1):
                values = series.iloc[i-len(weights):i].values
                diff_value = np.dot(weights, values)
                diff_series.append(diff_value)

            diffed_data[col] = diff_series

    return pd.DataFrame(diffed_data, index=dataframe.index[len(weights)-1:])
'''
def fractional_diff(series, differencing_value=0.1, threshold=1e-5):
    """
    Returns the fractionally differenced series.
    """
    series =series.copy()
    series= series['Open']
    # Determine the weights
    weights = get_weights(differencing_value, series.shape[0])
    print(weights)
    # Ensure weights are above the threshold
    weights = weights[np.abs(weights) > threshold]
    
    # Fractionally difference using the computed weights
    diff_series = []
    for i in range(len(weights), series.shape[0]+1):
        values = series.iloc[i-len(weights):i].values
        values = np.array(values)  # Ensure values is a numpy array
        diff_value = np.dot(weights.T, values)
        diff_series.append(diff_value)
    
    return pd.Series(diff_series, index=series.iloc[len(weights)-1:].index)

#######Find the minimum D Value that passes the ADF test

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Step 1: Compute weights for fractional differencing
def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

# Step 2: Function for fractional differencing
def fractional_diff(series, d, threshold=1e-5):
    weights = get_weights(d, series.shape[0])
    weights = weights[np.abs(weights) > threshold]
    
    diff_series = []
    for i in range(len(weights), series.shape[0] + 1):
        values = series.iloc[i-len(weights):i].values
        diff_value = np.dot(weights.T, values)
        diff_series.append(diff_value)

    return pd.Series(diff_series, index=series.iloc[len(weights)-1:].index)

def is_constant(series):
    return series.nunique() == 1

# Step 3: Loop to find minimum d value that passes ADF test
def find_min_d(series, max_d=1, step=0.01):
    if is_constant(series):
        return None

    d = 0
    p_val = 1
    while d <= max_d and p_val > 0.05:
        diffed_series = fractional_diff(series, d)
        if not is_constant(diffed_series):
            p_val = adfuller(diffed_series, maxlag=1)[1]  # Using ADF test
            d += step
        else:
            return None

    return d if p_val <= 0.05 else None


def find_min_d_for_df(df, max_d=1, step=0.01):
    d_values = {}
    df = df.copy()
    df = df.drop(['Date'], axis=1)
    df= df[150:]
   # print(df.columns)
#    return df.columns
 
    for column in df.columns:
        series = df[column]
        min_d = find_min_d(series, max_d, step)
        d_values[column] = min_d
    return d_values

