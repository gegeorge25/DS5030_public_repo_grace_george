#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%


def k(u):
    return np.exp(-u**2)/np.sqrt(2*np.pi)

def lcls(x,y,h=None,plot=True):
    grid=np.sort(np.unique(x))
       #k=1/np.sqrt(2*np.pi)
    if h is None:
        h=1.06 * np.std(x)* len(x) **(-0.2)

    y_hat = []    
    for z in grid:
        num=0
        dem=0
        for i in range(len(x)):
            u = (x[i] -z)/h
            num+=y[i] * k(u)
            dem+=k(u)
        y_hat_z=num/dem
        y_hat.append(y_hat_z)
        
    if plot:
        sns.scatterplot(data=gdf, y='WeightKg', x='WaistCircumferenceCm',alpha=.05)
        sns.lineplot(x=grid,y=y_hat, color='orange')

    # return y_hat, grid # Comment out return so output does not print out full list of y hats
    
    #%%
    