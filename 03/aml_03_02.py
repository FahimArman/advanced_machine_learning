import numpy as np

#generate training data

x1= np.random.multivariate_normal(mean = [1, 1], cov = [[1,0],[0,1]], size = 100)
#print(x1)
x2= np.random.multivariate_normal(mean = [3, 1], cov = [[1,0],[0,1]], size = 100)
x3= np.random.multivariate_normal(mean = [2, 2], cov = [[1,0],[0,1]], size = 100)
X = np.vstack([x1, x2, x3])

y = np.concatenate([np.repeat(0,200), np.repeat(1,100)])

from scipy.stats import multivariate_normal
d1 = multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]])
d2 = multivariate_normal(mean=[3,1], cov=[[1,0],[0,1]])
d3 = multivariate_normal(mean=[2,2], cov=[[1,0],[0,1]])
p1 = d1.pdf(X)
p2 = d2.pdf(X)
p3 = d3.pdf(X)
pxC0 = 0.5*p1 + 0.5*p2
pxC1 = p3

pY1 = (pxC1/3)/(pxC0*2/3 + pxC1/3)

pY0 = (pxC0*2/3)/(pxC0*2/3 + pxC1/3)

np.log(pY1/pY0)

np.round(pY1)

np.mean(np.round(pY1) == y)
