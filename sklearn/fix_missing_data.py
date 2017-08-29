import numpy as np
from sklearn.preprocessing import Imputer

# The following snippet demonstrates how to replace missing values, encoded as np.nan, 
# using the mean value of the columns (axis 0) that contain the missing values:

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))   
print('==========end============')
# The Imputer class also supports sparse matrices:

import scipy.sparse as sp
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
print(X)
print('======================')
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit(X)

X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
print(X_test)
print('======================')
print(imp.transform(X_test))
