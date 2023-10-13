import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta






def time_bars(raw_data):
    
    df =raw_data
    df['Daily_Returns'] = df['Close'].pct_change()
    df = df[:]

    
    return df
'''
def get_volume_bars(ohlc_df, volume_threshold):
    """
    Create volume bars from OHLC data.

    Parameters:
    - ohlc_df: DataFrame with OHLC and volume data.
    - volume_threshold: Volume threshold to sample bars.

    Returns:
    - DataFrame of volume bars.
    """
    # Calculate cumulative volume
    cum_volume = ohlc_df['Volume'].cumsum()

    # This will be our running total of thresholds
    threshold = volume_threshold

    # This function will check if the cumulative volume is greater than our running threshold
    def check_threshold(vol):
        nonlocal threshold
        if vol >= threshold:
            threshold += volume_threshold
            return True
        return False

    # Apply our function to each cumulative volume
    idx = cum_volume.apply(check_threshold)

    # Sample the original dataframe at indices where the threshold is crossed
    volume_bars_df = ohlc_df.loc[idx].copy()

    return volume_bars_df
'''

def get_volume_bars(ohlc_df, lookback_period):
    """
    Create volume bars from OHLC data using adaptive threshold based on rolling mean.

    Parameters:
    - ohlc_df: DataFrame with OHLC and volume data.
    - lookback_period: Number of periods to consider for rolling mean.

    Returns:
    - DataFrame of volume bars.
    """
    # Calculate rolling mean of volume
    rolling_mean = ohlc_df['Volume'].rolling(window=lookback_period).mean().shift(1)
    cum_volume = ohlc_df['Volume'].cumsum()

    # The threshold is the running total of rolling means
    threshold = rolling_mean.cumsum()
    #threshold = 5000
    # Compare cumulative volume with the rolling threshold
    idx = cum_volume >= threshold

    # Sample the original dataframe at indices where the threshold is crossed
    volume_bars_df = ohlc_df.loc[idx].copy()

    return volume_bars_df

def get_dollar_bars(time_bars, dollar_threshold):
    
    time_bars = time_bars.to_dict('records') 

    # Initialize an empty list of dollar bars
    dollar_bars = []

    # Initialize the running dollar volume at zero
    running_volume = 0

    # Initialize the running high and low with placeholder values
    running_high, running_low = 0, math.inf

    # For each time bar...
    for i in range(len(time_bars)):

        # Get the timestamp, open, high, low, close, and volume of the next bar
        next_close, next_high, next_low, next_open, next_timestamp, next_volume = [time_bars[i][k] for k in ['Close', 'High', 'Low', 'Open', 'Date', 'Volume']]

        # Assuming next_timestamp is your UNIX timestamp
        #next_timestamp_dt = datetime.fromtimestamp(next_timestamp, "Y-%m-%d")
#        next_timestamp = str(next_timestamp)
        next_timestamp_dt = datetime.utcfromtimestamp(next_timestamp)

        next_timestamp_dt = next_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')
        # Convert the string timestamp to a datetime object
        #next_timestamp_dt = datetime.strptime(next_timestamp, "%Y-%m-%d")

        # Get the midpoint price of the next bar (the average of the open and the close)
        midpoint_price = ((next_open) + (next_close))/2

        # Get the approximate dollar volume of the bar using the volume and the midpoint price
        dollar_volume = next_volume * midpoint_price

        # Update the running high and low
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)

        # If the next bar's dollar volume would take us over the threshold...
        if dollar_volume + running_volume >= dollar_threshold:

            # Set the timestamp for the dollar bar
            bar_timestamp = next_timestamp_dt + timedelta(minutes=1)
            
            # Convert the datetime object to the desired format
            bar_timestamp_str = bar_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Add a new dollar bar to the list of dollar bars
            dollar_bars += [{'Date': bar_timestamp_str, 'Open': next_open, 'High': running_high, 'Low': running_low, 'Close': next_close}]

            # Reset the running volume to zero
            running_volume = 0

            # Reset the running high and low to placeholder values
            running_high, running_low = 0, math.inf

        # Otherwise, increment the running volume
        else:
            running_volume += dollar_volume

    # Return the list of dollar bars

    dollar_bars = pd.DataFrame.from_dict(dollar_bars)
    #####################################################  Add percent change column to dollar bar DF. 

    dollar_bars['Daily_Returns'] = dollar_bars['Close'].pct_change()


    return dollar_bars



def dollar_bars(ohlc_df, dollar_threshold):
    """
    Create dollar bars from OHLC data.

    Parameters:
    - ohlc_df: DataFrame with OHLC and volume data.
    - dollar_threshold: Dollar threshold to sample bars (price * volume).

    Returns:
    - DataFrame of dollar bars.
    """
    # Calculate the dollar value for each row
    ohlc_df['DollarValue'] = ohlc_df['Close'] * ohlc_df['Volume']

    # Calculate cumulative dollar value
    cum_dollar_value = ohlc_df['DollarValue'].cumsum()

    # This will be our running total of thresholds
    threshold = dollar_threshold

    # This function will check if the cumulative dollar value is greater than our running threshold
    def check_threshold(value):
        nonlocal threshold
        if value >= threshold:
            threshold += dollar_threshold
            return True
        return False

    # Apply our function to each cumulative dollar value
    idx = cum_dollar_value.apply(check_threshold)

    # Sample the original dataframe at indices where the threshold is crossed
    dollar_bars_df = ohlc_df.loc[idx].copy()

    return dollar_bars_df
