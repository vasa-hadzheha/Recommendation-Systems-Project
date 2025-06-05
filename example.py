import numpy as np

# Given matrix
matrix = np.array([[1,1,1,0,0],
                   [3,3,3,0,0],
                   [4,4,4,0,0],
                   [5,5,5,0,0],
                   [0,2,0,4,4]])


import numpy as np

# Given matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Perform SVD
U, sigma, Vt = np.linalg.svd(matrix)

# Construct Σ matrix
Sigma = np.zeros(matrix.shape)
Sigma[:matrix.shape[0], :matrix.shape[1]] = np.diag(sigma)

# Reconstruct the original matrix using U, Σ, and Vt
reconstructed_matrix = np.dot(U, np.dot(Sigma, Vt))

# Print U, Σ, Vt, and their dot product
print('U:\n', U)
print('\nΣ:\n', Sigma)
print('\nV.T:\n', Vt)
print('\nU * Σ * V.T:\n', reconstructed_matrix)


