import pandas as pd
from math import norm

def bet_size_from_probability(P):
    """
    Calculate the bet size based on predicted probability for a binary symmetric outcome.
    
    Args:
    - P (float): Predicted probability of winning the bet.
    
    Returns:
    - float: Bet size as a fraction of capital.
    """
    return 2 * P - 1

def getsignal(events, stepSize, prob, pred, numClasses, numThreads, **kwargs):
    if prob.shape[0] == 0: return pd.Series()
    signal10= (prob -1 / numClasses/ (prob * (prob*(1-prob)) **.5 ))
    signal10= pred (2*norm.cdf(signal10)-1)
    #if 'side' in events: signal*= events.loc[signal10.indexm 'side']
    df0 = signal10.to_frame('signal').join(events[['t1']], how='left')
    df0= avgActiveSignals(df=0 , stepsize=stepSize)
    signal1 = discreetSignal(signal10 = df0, stepSize= stepSize)
    return signal10



    




