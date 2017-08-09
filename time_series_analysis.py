'''
Autoregressive Moving Average (ARMA): Sunspots data
'''
from __future__ import print_function

from operator import itemgetter
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
def tsa_local_min(indices, data):
	local_index = []
	local_value = []
	for index, entry in enumerate(data):	
		if (index > 0 and index < (data.shape[0]- 1)):
			if ((data[index] <= data[index - 1]) and (data[index] <= data[index + 1])):
				local_index.append(indices[index])
				local_value.append(data[index])	
		if (index == 0):
			if (data[index] <= data[index + 1]):
				local_index.append(indices[index])
				local_value.append(data[index])
		if (index == (data.shape[0]- 1)):
			if (data[index, 0] <= data[index - 1, 0]):
				local_index.append(indices[index])
				local_value.append(data[index])
	return local_index, local_value


def tsa_local_max(indices, data):
	local_index = []
	local_value = []
	for index, entry in enumerate(data):	
		if (index > 0 and index < (data.shape[0]- 1)):
			if ((data[index] >= data[index - 1]) and (data[index] >= data[index + 1])):
				local_index.append(indices[index])
				local_value.append(data[index])	
		if (index == 0):
			if (data[index] >= data[index + 1]):
				local_index.append(indices[index])
				local_value.append(data[index])
		if (index == (data.shape[0]- 1)):
			if (data[index, 0] >= data[index - 1, 0]):
				local_index.append(indices[index])
				local_value.append(data[index])
	return local_index, local_value

def tsa_merge_array_by_time(array1_index, array1_value, array2_index, array2_value):
	array_index = array1_index + array2_index
	tmp = array1_value + array2_value
	array_value = []
	indices, array_index = zip(*sorted(enumerate(array_index), key=itemgetter(1)))
	for i, entry in enumerate(indices):
		array_value.append(tmp[entry])
	return array_index, array_value





print(sm.datasets.sunspots.NOTE)
dta = sm.datasets.sunspots.load_pandas().data
# print(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))

del dta["YEAR"]

local_minimum_index, local_minimum_value = tsa_local_min(dta.index, dta.values)
local_maximum_index, local_maximum_value = tsa_local_max(dta.index, dta.values)
local_optimum_index, local_optimum_value = tsa_merge_array_by_time(local_minimum_index, local_minimum_value, 
																	local_maximum_index, local_maximum_value)


plt.figure('original data')
plt.plot(dta.index, dta.SUNACTIVITY)
# plt.scatter(local_minimum_index, local_minimum_value)
# plt.scatter(local_maximum_index, local_maximum_value)
plt.plot(local_optimum_index, local_optimum_value)



plt.show()

