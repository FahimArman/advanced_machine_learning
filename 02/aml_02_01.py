def dash(a=20):
	print("-" * a)

import numpy as np

#Matrix
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Matrix: \n", A)
print("Shape is: ", A.shape)
dash()

#vector
v1 = np.array([[1,2,3]])
print("vector is: ", v1)
print("shape is: ", v1.shape)
dash()

#vector
v2 = np.array([[1],[2],[3]])
print("vector is: ", v2)
print("shape is: ", v2.shape)
dash()


#vector and matrix combination

A2 = np.vstack([v1,v1])
print(A2)
A3 = np.hstack([v2,v2])
print(A3)
dash()

A4 = np.hstack([A,A])
print(A4)
dash()

A5 = np.vstack([A,A])
print(A5)
dash()


#subvectors

print(A4[0,:])
print(A4[:,0])
print(A4[3:4,:])
print(A4[1:2,:])
print(A4[1:3,:])
print(A4[:,1:3])
print(A4[1:3,1:3])
dash()

z = np.zeros((3,2))
print(z)
dash()

o = np.ones((4,2))
print(o)
dash()

f = np.full(5,5)
print(f)
dash()

e = np.eye(6)
print(e)
dash()

d = np.diag([1,2,3])
print(d)
dash()


#Challenge
"""
Create a 10 × 10 matrix 
where all elements are 5, 
except the diagonal is 0.
"""
print("Challenge: \n")
P = np.ones((10,10))*5
Q = np.eye(10)*5
R = P - Q 

print(R)

# Matrix product

M = A.dot(v2)
print(M)

s = v1.dot(v2)
print(s)

At = A.T
print(At)

v2t = v2.T
print(v2t)


v2tv2 = v2.T.dot(v2)
print(v2tv2)

Av2 = A.dot(v2)
print(Av2)

AAt = A.dot(A.T)
print(AAt)

# Elementwise Multiplication
A_At = np.multiply(A,A.T)
print(A_At)

A_v2 = np.multiply(A,v2)
print(A_v2)

#Matrix determinant
detA = np.linalg.det(A)
print(detA)

B =np.array([[1,2,3],[4,5,6],[7,8,13]])

print("Inverse of A = ", np.linalg.inv(A))
print("Inverse of B = ", np.linalg.inv(B))


eigvals, eigvecs = np.linalg.eig(B)
print("eigenvals = ", eigvals, "\neigenvectors = ", eigvecs)


U, S, Vtranspose = np.linalg.svd(A4)
print("U = ", U, "\nS = ", S, "\nV' = ", Vtranspose)


