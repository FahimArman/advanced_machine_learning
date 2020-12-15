import numpy as np

#A = np.array([[56,85,9],[70,56,56],[27,100,50]])

A = np.array([[58,96,27],[51,81,45],[67,4,19]])

A_inv = np.linalg.inv(A)
dotRR1=np.dot(A,A_inv)

print(np.trace(dotRR1))
