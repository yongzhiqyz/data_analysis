# http://blog.csdn.net/lanchunhui/article/details/50521648

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据集
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'+
	'breast-cancer-wisconsin/wdbc.data', header=None)
# Breast Cancer Wisconsin dataset

X, y = df.values[:, 2:], df.values[:, 1]
                                # y为字符型标签
                                # 使用LabelEncoder类将其转换为0开始的数值型
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
print('X_train: ', X_train.shape)
print('type of the X_train: ', type(X_train))
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('the head 5 of the x_train: ', X_train[0:5,:])
print('the head 5 of the y_train: ', y_train[0:5])

# 可放在Pipeline中的步骤可能有：
# 特征标准化是需要的，可作为第一个环节
# 既然是分类器，classifier也是少不了的，自然是最后一个环节
# 中间可加上比如数据降维（PCA）

# Pipeline对象接受二元tuple构成的list，

# 每一个二元 tuple 中的第一个元素为 arbitrary identifier string，
# 我们用以获取（access）Pipeline object 中的 individual elements，

# 二元 tuple 中的第二个元素是 scikit-learn与之相适配的transformer 或者 estimator。

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))
                    ])
pipe_lr.fit(X_train, y_train)
print('Test accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# Pipeline 的中间过程由scikit-learn相适配的转换器（transformer）构成，
# 最后一步是一个estimator。比如上述的代码，StandardScaler和PCA transformer 
# 构成intermediate steps，LogisticRegression 作为最终的estimator。

# 当我们执行 pipe_lr.fit(X_train, y_train)时，
# 首先由StandardScaler在训练集上执行 fit和transform方法，
# transformed后的数据又被传递给Pipeline对象的下一步，也即PCA()。
# 和StandardScaler一样，PCA也是执行fit和transform方法，
# 最终将转换后的数据传递给 LosigsticRegression。