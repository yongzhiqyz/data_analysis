import numpy as np
import pandas as pd
from pandas.tools import plotting
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols

data = pd.read_csv('brain_size.csv', sep = ';', na_values= '.')
#print data.head(5)

# print type(data)
# print '=============================================='
# print data.Gender, data.Weight

t = np.arange(-np.pi, np.pi, 0.01)
sin_t = np.sin(t)
cos_t = np.cos(t)

dta = pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})
#print '=========================================='
#print dta.head(5)
#
#print dta.shape
#print dta.columns
#print dta[dta.columns[0]]

groupby_gender = data.groupby('Gender')
print groupby_gender.max()
print '========================='
for gender, value in groupby_gender['Weight']:
    print((gender, value.mean()))

groupby_gender.boxplot(column=['FSIQ', 'VIQ', 'PIQ'])

# Scatter matrices for different columns
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])

Ts, p_value = stats.ttest_1samp(data['VIQ'], 0) 

one_sample_results = stats.ttest_1samp(data['VIQ'], 0) 
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
two_sample_results = stats.ttest_ind(female_viq, male_viq) 
difference_between_FSIQ_PIQ0 = stats.ttest_ind(data['FSIQ'], data['PIQ']) 
#The problem with this approach is that it forgets that there are links 
#between observations: FSIQ and PIQ are measured on the same individuals. 
#Thus the variance due to inter-subject variability is confounding, 
#and can be removed, using a “paired test”, or “repeated measures test”:

difference_between_FSIQ_PIQ = stats.ttest_rel(data['FSIQ'], data['PIQ'])  

#T-tests assume Gaussian errors. We can use a Wilcoxon signed-rank test, 
#that relaxes this assumption:

wilcoxon_test_result = stats.wilcoxon(data['FSIQ'], data['PIQ'])   
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary())
data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)

model = ols("iq ~ type", data_long).fit()
print(model.summary())

# create data and export data to csv file
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
df.to_csv('births1880.csv',index=False,header=True)
df.to_csv('births1880.txt',index=False,header=True)
#df.to_excel('births1880.xls')

plt.show()


data = pd.read_csv('examples/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary())



