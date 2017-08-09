"""
One-class SVM is an unsupervised algorithm that learns a decision function for novelty detection: 
classifying new data as similar or different to the training set.
One-class SVM is used for novelty detection, that is, given a set of samples, 
it will detect the soft boundary of that set so as to classify new points as belonging to that set or not. 
The class that implements this is called OneClassSVM.
In this case, as it is a type of unsupervised learning, the fit method will only take as input an array X, 
as there are no class labels.
See, section Novelty and Outlier Detection for more details on this usage.

"""


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
print ('the size of xx: ', xx.shape)
print ('the size of yy: ', yy.shape)

# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
print ('the size of X_train: ', X_train.shape)
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
print ('the size of X_test: ', X_test.shape)
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
print ('the size of X_outliers: ', X_outliers.shape)

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
print ('the size of xx.ravel(): ', xx.ravel().shape)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
print ('the size of Z:', Z.shape)
Z = Z.reshape(xx.shape)
print ('the reshaped size of Z: ', Z.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap=plt.cm.PuBu)
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10), cmap=plt.cm.Oranges)
print ('the min of Zmin: ', Z.min(), 'Zmax:', Z.max())
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='black')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
