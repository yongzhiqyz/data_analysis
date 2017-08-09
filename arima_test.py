import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO

wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
print ('the first 10 records of wpi1: ', wpi1[:10])
data = pd.read_stata(BytesIO(wpi1))
print '============='
print  data
data.index = data.t



# Fit the model
mod = sm.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(1,1,1))
res = mod.fit(disp=False)
print(res.summary())





plt.figure('SARIMAX')
plt.plot(data.t, data.wpi)
plt.show()


