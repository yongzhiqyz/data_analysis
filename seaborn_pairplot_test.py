import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris)

#Use different markers for each level of the hue variable:
#Use a different color palette:
#Show different levels of a categorical variable by the color of plot elements:
g = sns.pairplot(iris, hue="species", palette="husl", markers=["o", "s", "D"])

#Draw larger plots and a subset of variables:
g = sns.pairplot(iris, size=3, vars=["sepal_width", "sepal_length"])

#Plot different variables in the rows and columns:
g = sns.pairplot(iris, x_vars=["sepal_width", "sepal_length"], y_vars=["petal_width", "petal_length"])

#Use kernel density estimates for univariate plots:
g = sns.pairplot(iris, diag_kind="kde")

#Fit linear regression models to the scatter plots
g = sns.pairplot(iris, kind="reg")

#Pass keyword arguments down to the underlying functions (it may be easier to use PairGrid directly):
g = sns.pairplot(iris, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))

plt.show()


