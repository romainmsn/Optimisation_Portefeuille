import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


#import data
def GetStocksPrice(stocks,start,end):
    stocks =pdr.get_data_yahoo(stocks,start=start,end=end)
    stocks = pd.DataFrame(stocks['Close'])
    return stocks 

#concat 
def concatener_series_temporelles(*series_temporelles):
    return pd.concat(series_temporelles,axis=1)


start = datetime.datetime(2018,1,1)
end = datetime.datetime(2022,1,1)

nombre_actions = int(input("Entrez le  nombre d'actions souhaité : "))

nom_actions = []

for i in range(nombre_actions):
    nom_action = input(f'''entrez le nom de l'action {i+1} :''')
    nom_actions.append(nom_action)

series_temporelles = [GetStocksPrice(action, start, end) for action in nom_actions]
resultat_concatenation = concatener_series_temporelles(*series_temporelles)

print(resultat_concatenation)

