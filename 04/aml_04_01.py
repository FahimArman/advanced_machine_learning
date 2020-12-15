import numpy as np
from sklearn import datasets

iris = datasets.load_iris() 
X = iris.data
y = iris.target

[N, Di] = X.shape #150,4
Do = 3 
H1 = 10
alpha = 0.00001
y = np.reshape(y, (N,1))

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)

np.random.seed(1)
w2 = np.random.randn(Di, H1) #(4*10)
w1 = np.random.randn(H1, Do) #(10*3)


def linearLoss(y, yhat):
	return(np.square(yhat - y).sum())
def linearLossGrad(y, yhat):
	return(2*(yhat - y))

lossFunction = linearLoss
lossGrad = linearLossGrad

def relu(x):
	return(np.maximum(x, 0))
def reluGrad(x):
	if (x < 0):
		return(0)
	return(1)

activation = np.vectorize(relu)
activationGrad = np.vectorize(reluGrad)

def accuracy(y, yhat):
    zhat = np.apply_along_axis(np.argmax, 1, yhat)
    z = np.apply_along_axis(np.argmax, 1, y)
    return(np.mean(zhat == z))

for iteration in range(10000):

    h1 = X.dot(w2) 
    fh1 = activation(h1)
    yhat = fh1.dot(w1)
    
    loss = lossFunction(y, yhat)

    print('[ iter:', iteration, '] loss:', loss, end='\r', flush=True)

    dJdy = lossGrad(y, yhat) 
    dydw1 = fh1
    dJdw1 = dydw1.T.dot(dJdy)


    dydh = w1
    dhdg = activationGrad(h1)
    dgdw2 = X
    dJdh = dJdy.dot(dydh.T)
    dJdg = dJdh*dhdg
    dJdw2 = dgdw2.T.dot(dJdg)
    
    w1 = w1 - alpha*dJdw1
    w2 = w2 - alpha*dJdw2


print('\n')

print("Training accuracy:", accuracy(y, yhat))
