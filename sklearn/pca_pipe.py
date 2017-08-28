# coding:utf-8
# http://www.cnblogs.com/taceywong/p/5931253.html

# 我们使用GridSearchCV来设置PCA的维度
from pylab import *
import numpy as np

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target


# 绘制PCA图谱
myfont = matplotlib.font_manager.FontProperties(fname="Microsoft-Yahei-UI-Light.ttc")
mpl.rcParams['axes.unicode_minus'] = False
pca.fit(X_digits)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel(u'n_components',fontproperties=myfont)
plt.ylabel(u'解释方差',fontproperties=myfont)
plt.title(u"主成分分析谱",fontproperties=myfont)


# 预测
plt.clf()
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_digits, y_digits)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=myfont)
plt.title(u"预测",fontproperties=myfont)