import numpy as np
import pandas as pd
from sklearn.utils import resample

def sequential_bootstrap(data, sample_size, window_size):
    """
    Generate a sample using sequential bootstrapping.
    
    Parameters:
    - data: the time series data (as a DataFrame or Series)
    - sample_size: desired size of the bootstrapped sample
    - window_size: size of the sampling window
    
    Returns:
    - A bootstrapped sample
    """
    assert sample_size >= window_size, "Sample size should be >= window size."
    
    n = len(data)
    sample = []
    
    for i in range(sample_size):
        # Start of the window
        start = max(0, i - window_size)
        
        # Draw a random index from the window
        idx = np.random.randint(start, min(start + window_size, n))
        sample.append(data.iloc[idx])
    
    return pd.DataFrame(sample)

# Example usage:
# Assuming `df` is your time series data
# Here, we're drawing a sample of size 1000 using a window of 10
#bootstrapped_sample = sequential_bootstrap(df, sample_size=1000, window_size=10)
import numpy as np
import pandas as pd

def get_probabilities(data, target_col):
    """
    Calculate sampling probabilities for each class in a way that under-represented classes 
    have higher probability.
    """
    class_counts = data[target_col].value_counts()
    max_count = class_counts.max()
    probabilities = max_count / class_counts
    probabilities = probabilities / probabilities.sum()
    return probabilities

def sequential_bootstrap_with_rebalance(data, target_col='label', sample_size=1000, window_size=10):
    """Sequential bootstrap sampling with a focus on rebalancing the labels."""
    sample_indices = []

   
    indices = list(data.index)
    current_index_position = np.random.choice(len(indices))
    label_probabilities = get_probabilities(data, target_col)
    
    for _ in range(sample_size):
        sample_indices.append(indices[current_index_position])
        start_pos = max(0, current_index_position - window_size)
        end_pos = min(len(indices) - 1, current_index_position + window_size)
        eligible_data = data.iloc[start_pos:end_pos]
        sample_probs = eligible_data[target_col].map(label_probabilities).values
        sample_probs = sample_probs / sample_probs.sum()
        current_index_position = np.random.choice(range(start_pos, end_pos), p=sample_probs)
    
    return data.loc[sample_indices]