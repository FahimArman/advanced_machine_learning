#import sklearn # we will use the sklearn package
#import sklearn.datasets

import numpy as np
import matplotlib.pyplot as plt


#generate training data

x1= np.random.multivariate_normal(mean = [1, 1], cov = [[1,0],[0,1]], size = 100)
#print(x1)
x2= np.random.multivariate_normal(mean = [3, 1], cov = [[1,0],[0,1]], size = 100)
x3= np.random.multivariate_normal(mean = [2, 2], cov = [[1,0],[0,1]], size = 100)
X = np.vstack([x1, x2, x3])
#print(X)
#print(x1.shape)
#print(x2.shape)
#print(x3.shape)
#print(X.shape)
#print(X)
#print("-" * 20)
y = np.concatenate([np.repeat(0,200), np.repeat(1,100)])

#print(y)
#plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
#plt.show()



##generate testing data, using the same distribution as the training data
x1= np.random.multivariate_normal(mean = [1, 1], cov= [[1,0],[0,1]], size = 100)
x2= np.random.multivariate_normal(mean = [3, 1], cov= [[1,0],[0,1]], size = 100)
x3= np.random.multivariate_normal(mean = [2, 2], cov= [[1,0],[0,1]], size = 100)

Xtest = np.vstack([x1, x2, x3])
ytest = np.concatenate([np.repeat(0,200), np.repeat(1,100)])
#print(Xtest)
#print(ytest)

from sklearn import svm
m = svm.SVC(kernel='poly', C=1, degree = 1, coef0 = 1)
m.fit(X, y)

print(m.score(X, y)) # training accuracy
print(m.score(Xtest, ytest)) # test accuracy

C = 2 # must be positive
d = 3 # must be a positive integer
m = svm.SVC(kernel='poly', C=C, degree = d, coef0 = 1)
m.fit(X, y)
print("C =", C, ", degree =", d, ", Training:", m.score(X, y),", Testing:" , m.score(Xtest, ytest))



def trainAccuracy(C = 1):
	m = svm.SVC(kernel='poly', C=C, degree = 2, coef0 = 1)
	m.fit(X, y)
	print("C =", C, ", degree = 2, Training:", m.score(X, y))
	return(m.score(X, y))

def testAccuracy(C = 1):
	m = svm.SVC(kernel='poly', C=C, degree = 2, coef0 = 1)
	m.fit(X, y)
	print("C =", C,", degree =2, Testing:", m.score(Xtest, ytest))
	return(m.score(Xtest, ytest))

Cset = 2.0**np.arange(-5,5)
print(Cset)
trainAcc = [trainAccuracy(C = C) for C in Cset]
testAcc = [testAccuracy(C = C) for C in Cset]


#plt.xscale('log')
#plt.scatter(Cset, trainAcc, c = 'k', label = "Train")
#plt.scatter(Cset, testAcc, c = 'r', label = "Test")
#plt.legend()
#plt.show()


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


kf = KFold(n_splits=10)
m = svm.SVC(kernel='poly', C=1, degree = 1, coef0 = 1)
score = cross_val_score(m, X, y, cv=kf)
cvScore = score.mean()
print(cvScore)
m.fit(X, y)
print(m.score(Xtest, ytest))




def cvAccuracy(C = 1):
	kf = KFold(n_splits=10)
	m = svm.SVC(kernel='poly', C=C, degree = 2, coef0 = 1)
	score = cross_val_score(m, X, y, cv=kf)
	cvScore = score.mean()
	return(cvScore)

cvTrainAcc = [cvAccuracy(C = C) for C in Cset]

plt.xscale('log')
plt.scatter(Cset, trainAcc, c = 'k', label = "Train")
plt.scatter(Cset, testAcc, c = 'r', label = "Test")
plt.scatter(Cset, cvTrainAcc, c = 'b', label = "CV")
plt.legend()
plt.show()




print("-" * 20)
print("GridSearchCV")
print("-" * 20)
from sklearn.model_selection import GridSearchCV
## set the search to examine gamma from 2^-5 to 2^4 and C from 2^-5 to 2^4
parameterSet = [{'kernel': ['poly'], 'degree': np.arange(1,5),
'coef0' : [1], 'C': 2.0**np.arange(-5,5)}]
m = svm.SVC()
gsm = GridSearchCV(m, parameterSet, cv=5)
gsm.fit(X, y)

print(gsm.best_params_)
