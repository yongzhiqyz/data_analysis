# coding:utf-8
# http://www.cnblogs.com/taceywong/p/5930662.html
# 作者: Andreas Mueller <amueller@ais.uni-bonn.de>
# 协议: BSD 3 

# 在很多现实世界的例子中,有很多从数据集中提取特征的方法.很多时候我们需要结合多种方法获得好的效果.
# 本例将展示怎样使用FeatureUnion通过主成分分析和单变量选择相进行特征结合.

# 结合使用转换器的好处是它允许在整个过程中进行交叉验证和网格搜索。

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target

#本数据集维度较高,最好进行PCA降维
pca = PCA(n_components=2)

#也许一些原始特征也非常有用
selection = SelectKBest(k=1)

#从主成分分析和单变量选择的建立评估器
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#使用组合特征来转换数据集
#X_features = combined_features.fit(X, y).transform(X)


svm = SVC(kernel="linear")

#进行网格搜索(over k, n_components and C)
pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)

from sklearn.externals import joblib
joblib.dump(grid_search, 'filename.pkl') 
grid_search1 = joblib.load('filename.pkl') 
y_predict = grid_search1.predict(X)
print ('predict:  ', y_predict)

