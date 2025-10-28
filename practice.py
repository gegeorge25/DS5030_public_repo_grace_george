#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Parameters for exponential distribution
lambda_param = 1
x = np.linspace(0, 5, 500)

# PDF
pdf = lambda_param * np.exp(-lambda_param * x)

# CDF
cdf = 1 - np.exp(-lambda_param * x)

# Plot
plt.figure(figsize=(8,5))
plt.plot(x, pdf, label='PDF (f_X(x))', color='blue', linewidth=2)
plt.plot(x, cdf, label='CDF (F_X(x))', color='red', linewidth=2)
plt.fill_between(x, 0, pdf, color='blue', alpha=0.2)  # highlight density under PDF
plt.title('Exponential Distribution: PDF and CDF')
plt.xlabel('x')
plt.ylabel('Probability / Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()

    
    #%%
 

# Define x values
x = np.linspace(0, 1, 500)

# PDF of uniform[0,1]
pdf = np.ones_like(x)

# CDF of uniform[0,1]
cdf = x

# Plot
plt.figure(figsize=(8,5))
plt.plot(x, pdf, label='PDF', color='blue', linewidth=2)
plt.plot(x, cdf, label='CDF', color='red', linewidth=2)
plt.fill_between(x, 0, pdf, color='blue', alpha=0.2)  # shade area under PDF
plt.title('Uniform Distribution [0,1]: PDF and CDF')
plt.xlabel('x')
plt.ylabel('Probability / Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()

# %%
