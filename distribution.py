"""
different probability distribution

"""
print __doc__
import numpy as np
import matplotlib.pyplot as plt


# geometric distribution test
n_num = 10
k = np.linspace(1, n_num, n_num)
p = 0.5
# prob = [(np.power((1 - p), k[i]) * p) for i, data in enumerate(k)]
prob = [((1 - p) ** k[i]) * p for i, data in enumerate(k)]
print ('the size of k: ', k.shape)
# print ('the size of prob:', prob)
plt.figure('Geometric distribution')
plt.title('Geometric distribution')
plt.xlabel('testing number')
plt.ylabel('the probability')
plt.plot(k, prob)


# gauss distribution test
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
plt.figure('gauss distribution')
count, bins, ignored = plt.hist(s, 30, normed=True)
print ('bins: ',bins)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
          linewidth=2, color='r')
plt.show()
