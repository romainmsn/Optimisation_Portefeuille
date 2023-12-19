import pandas as pd 
import numpy as np
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

#import data
def GetStocksPrice(stocks,start,end):


