import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from scipy.optimize import minimize

#import data
start = datetime.datetime(2018,1,1)
end = datetime.datetime(2022,1,1)

apple = pdr.get_data_yahoo('AAPL',start=start,end=end)
apple = pd.DataFrame(apple['Close'])
amazon = pdr.get_data_yahoo('AMZN',start=start,end=end)
amazon = pd.DataFrame(amazon['Close'])
ibm = pdr.get_data_yahoo('IBM',start=start,end=end)
ibm = pd.DataFrame(ibm['Close'])
meta = pdr.get_data_yahoo('META',start=start,end=end)
meta = pd.DataFrame(meta['Close'])

stocks = pd.concat([apple,amazon,ibm,meta],axis=1)
stocks.columns = ('apple','amazon','ibm','meta')

log_ret = np.log(stocks/stocks.shift(1))

#optimisation méthode montecarlo
np.random.seed(667)

nbr_exp = 3000
all_weights = np.zeros([nbr_exp,len(stocks.columns)])
ret_arr = np.zeros(nbr_exp)
vol_arr = np.zeros(nbr_exp)
sharpe_arr = np.zeros(nbr_exp)

for ind in range(nbr_exp):
    weights = np.random.random(len(stocks.columns))
    weights = weights/np.sum(weights)

    all_weights[ind,:] = weights

    ret_arr[ind] = np.sum(log_ret.mean() * weights * 252)
    vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

ret_max = ret_arr[sharpe_arr.argmax()]
vol_max = vol_arr[sharpe_arr.argmax()]

print("Méthode de Monte Carlo")
print(f"L'allocation optimeale est : {all_weights[sharpe_arr.argmax()]}.")
print(f"Le ratio de sharpe maximal est : {sharpe_arr.max()}.")

#représentation graphique
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='inferno',s=30)
plt.colorbar(label='Ratio de Sharpe')
plt.xlabel('Volatilité')
plt.ylabel('Rendement')
plt.title('Simulation Monte Carlo')
plt.scatter(vol_max,ret_max,c='red',edgecolors='black',s=50)
plt.show()

#optimisation avec minimisation des moindres carrés
def get_ret_vol_sr(weights):
    ret = np.sum(log_ret.mean()*weights*252)
    vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def check_sum(weights):
    return np.sum(weights) - 1

def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * (-1)

guess_init = np.ones(len(log_ret.columns))/len(stocks.columns)
bounds = tuple((0,1) for i in range(len(log_ret.columns)))
cons = ({'type':'eq','fun':check_sum})  

opt_result = minimize(neg_sharpe,guess_init,method='SLSQP',bounds=bounds,constraints=cons)

print("Méthode de minimisation des moindres carrés : ")
print(f"L'allocation maximale est : {opt_result.x}.")
print(f"Le ratio de sharpe maximal est : {opt_result.fun * (-1)}.")

#Frontière efficiente
frontier_y = np.linspace(ret_arr.min(),ret_arr.max(),100)
frontier_vol = []

def vol_min(weights):
    return get_ret_vol_sr(weights)[1]


for possible_return in frontier_y:
    cons_2 = ({'type':'eq','fun':check_sum},{'type':'eq','fun':lambda w:get_ret_vol_sr(w)[0] - possible_return})
    result = minimize(vol_min,guess_init,method='SLSQP',bounds=bounds,constraints=cons_2)
    frontier_vol.append(result.fun)

#représentation graphique de la frontière efficiente 
plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='inferno',s=30)
plt.colorbar(label='Ratio de Sharpe')
plt.xlabel('Volatilité')
plt.ylabel('Rendement')
plt.title('Simulation de markowitz et frontière efficiente')
plt.plot(frontier_vol,frontier_y,'g--',linewidth=3)
plt.show()


#VaR 

alloc_opti = opt_result.x
daily_return = stocks.pct_change()
daily_return = daily_return.dropna()
daily_return_weighted = (daily_return*alloc_opti).sum(axis=1)
sorted_values = daily_return_weighted.sort_values(ascending=False)

level_of_confidence = 95
percentile = 100 - level_of_confidence
var_95 = np.percentile(sorted_returns, percentile)

print(f'Il y a 5% de chance que la perte journalière du portefeuille soit supérieure à {round(var_95,2)*(-1)*100} %')
