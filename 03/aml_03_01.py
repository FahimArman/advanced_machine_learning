import sklearn # we will use the sklearn package
import sklearn.datasets

import numpy as np
import matplotlib.pyplot as plt


#generate training data

x1= np.random.multivariate_normal(mean = [1, 1], cov = [[1,0],[0,1]], size = 100)
print(x1)
x2= np.random.multivariate_normal(mean = [3, 1], cov = [[1,0],[0,1]], size = 100)
x3= np.random.multivariate_normal(mean = [2, 2], cov = [[1,0],[0,1]], size = 100)
X = np.vstack([x1, x2, x3])
y = np.concatenate([np.repeat(0,200), np.repeat(1,100)])
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()

