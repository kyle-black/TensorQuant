import plotly.graph_objects as go


import pandas as pd
from datetime import datetime


def create_figure(df_dollar,df_time):
    fig_dollar = go.Figure(data=[go.Candlestick(x=df_dollar['Date'],
                    open=df_dollar['Open'],
                    high=df_dollar['High'],
                    low=df_dollar['Low'],
                    close=df_dollar['Close'],

                    
                    )])
    

    fig_time = go.Figure(data=[go.Candlestick(x=df_time['Date'],
                    open=df_time['Open'],
                    high=df_time['High'],
                    low=df_time['Low'],
                    close=df_time['Close'])])
    

    
    return fig_dollar, fig_time
    #fig.show()