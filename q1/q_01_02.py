import numpy as np

#A = np.array([[70,72,43],[35,73,37],[29,58,56]])
#B = np.array([[50,1,37],[58,28,97],[48,98,81]])

A = np.array([[51,56,56],[43,65,83],[54,5,70]])
B = np.array([[34,7,38],[93,45,27],[91,4,32]])

dotAB=np.dot(A,B)

print(np.trace(dotAB))
