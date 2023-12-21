import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from scipy.optimize import minimize

nombre_actions = int(input('''Quelle est le nombre d'action souhaité : '''))
start = input('Date de départ (YYYY-MM-DD) : ')
end = input('Date de fin (YYYY-MM-DD) : ')

#import les datas sur le prix des actions 
def GetStocksData(stock,start,end):
    stock = pdr.get_data_yahoo(stock,start=start,end=end)
    stock = pd.DataFrame(stock['Close'])
    return(stock)

stocks_arr = pd.DataFrame()
nom_actions = []

#Concaténation pour obtention d'un DataFrame des prix des différentes actions à la cloture
for i in range(nombre_actions):
    nom_action = str(input(f"Nom de la {i+1}ème action : "))
    stocks_arr[nom_action] = GetStocksData(nom_action,start,end)
    
print(stocks_arr.head())

#optimisation et algorithme de Sharpe avec méthode de Monte Carlo
np.random.seed(1)

log_ret = np.log(stocks_arr/stocks_arr.shift(1))

nbr_exp = int(input('''nombre d'expérience souhaitée pour la simulation de Monte Carlo : '''))
all_weights = np.zeros([nbr_exp,nombre_actions])
ret_arr = np.zeros(nbr_exp)
vol_arr = np.zeros(nbr_exp)
sharpe_arr = np.zeros(nbr_exp)

for ind in range(nbr_exp):
    
    weights = np.array(np.random.random(nombre_actions))
    weights = weights/weights.sum()

    all_weights[ind,:] = weights

    ret_arr[ind] = np.sum(log_ret.mean() * weights * 252)
    vol_arr[ind] = np.sqrt((np.dot(weights.T,np.dot(log_ret.cov(),weights*252))))

    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

ret_max = ret_arr[sharpe_arr.argmax()]
vol_max = vol_arr[sharpe_arr.argmax()]
print("Optimisation avec méthode de Montecarlo : ")
print(f"La valeur du ratio de sharpe maximale pour {nbr_exp} expériences est : {sharpe_arr.max()}")
print(f"la répartition des poids optimales est : {all_weights[sharpe_arr.argmax(),:]} ")

#Représentation graphique
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr)
plt.colorbar(label='Ratio de Sharpe')
plt.scatter(vol_max,ret_max,c='red',edgecolors='black',s=50)
plt.show()


#optimisation avec la méthode de minimisation (moindres carrés)
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights * 252)
    vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
    sr = ret/vol
    return np.array([vol, ret, sr])

def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * (-1)

def check_sum(weights):
    #retourne 0 si la somme des poids vaut 1 
    return np.sum(weights) - 1

cons = ({'type' : 'eq','fun':check_sum})
bounds = tuple((0,1) for i in range(nombre_actions))
init_guess = np.ones(nombre_actions) * (1/nombre_actions)
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
print("optimisation avec la méthode des moindres carrés : ")
print("Allocation optimale des titres : ")
print(opt_results.x)
print("Valeurs maximales du ratio de Sharpe : ")
print(get_ret_vol_sr(opt_results.x)[2])

#Frontière efficiente (pf avec le meilleur rendement pour un risque défini)