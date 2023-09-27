import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

def compute_and_plot_acf(series, nlags=40):
    """
    Compute and plot the ACF for a given series.

    Parameters:
    - series: pandas Series or numpy array
        Time series data
    - nlags: int, optional
        Number of lags to be used in ACF computation

    Returns:
    - ACF values and a plot of the ACF
    """
    # Compute ACF values
    
    
    
    acf_values = acf(series, nlags=nlags, fft=True)  # Using FFT to compute the ACF
    
    # Plot ACF values
    plt.figure(figsize=(10, 5))
    plot_acf(series, lags=nlags)
    plt.title("Autocorrelation Function")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.show()
    plt.savefig('path_to_save_plot.png')
    
    return acf_values

# Example
# Load time series data
# data = pd.read_csv("path_to_your_time_series_data.csv")
# series = data["your_column_name"]
# acf_vals = compute_and_plot_acf(series)
