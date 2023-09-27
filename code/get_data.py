#!/usr/bin/env python

#try:
    # For Python 3.0 and later
from urllib.request import urlopen
#except ImportError:
    # Fall back to Python 2's urllib2
#    from urllib2 import urlopen

import certifi
import json
import pandas as pd

def get_json(symbol):
    
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey=3e17d2b777a13feee4c1243985cdc7c4")
    
    response = urlopen(url, cafile=certifi.where())
    
    data = response.read().decode("utf-8")
    x =json.loads(data)

    x = x['historical'][:300]
    x = json.dumps(x)

    price_df = pd.read_json(x)
    price_df.rename(columns={'close':'Close', 'open':'Open', 'high':'High', 'low':'Low', 'volume':'Volume', 'date':'Date'}, inplace=True)
    
    
    return price_df 

#url = ("https://financialmodelingprep.com/api/v3/historical-price-full/SPY?apikey=3e17d2b777a13feee4c1243985cdc7c4")
#x =get_jsonparsed_data('SPY')

#print(x)



