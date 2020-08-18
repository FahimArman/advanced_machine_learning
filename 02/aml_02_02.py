import numpy as np

Z = np.array([[5.0, -2, 3], [4, 1, 1], [0, -3, 2]])
y = np.array([3, 0, 1])


x = np.linalg.inv(Z).dot(y)

print(x)


import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def neuron(x, w):
	return(sigmoid(x.dot(w)))

x = np.array([1, 1, 1])
w = np.array([-1, 2, 3])
ans = neuron(x, w)
print(ans)
