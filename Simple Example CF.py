#Прості приклади SVD окремо кожної складової U Sigma Vt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Example data: User-item interactions (ratings)
data = pd.DataFrame({
    "Item1": [5, 0, 0, 0, 0],
    "Item2": [0, 0, 24, 4, 5],
    "Item3": [1, 0, 5, 0, 2],
    "Item4": [0, 3, 0, 0, 4],
})
print("Original User-Item Matrix:")
print(data)
# Convert the DataFrame to a sparse matrix with a floating-point data type
user_item_matrix = csr_matrix(data.values, dtype=np.float64)
print("\nCSR Matrix:")
print(user_item_matrix)
# The number of factors to factorize the user-item matrix
NUMBER_OF_FACTORS_MF = 2

# Perform matrix factorization using Truncated SVD
U, sigma, Vt = svds(user_item_matrix, k=NUMBER_OF_FACTORS_MF)

print("\nU:")
print(U)

print("\nSigma:")
print(sigma)

print("\nVt:")
print(Vt)

# Reconstruct the predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)

# Converting the reconstructed matrix back to a Pandas DataFrame
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns=data.columns, index=data.index)

# Display the original user-item matrix
print("\nOriginal User-Item Matrix:")
print(data)

# Display the reconstructed user-item matrix with predicted ratings
print("\nReconstructed User-Item Matrix with Predicted Ratings:")
print(cf_preds_df)

# Example matrix
# A = np.array([[1, 2, 3],
#               [4, 5, 6],
#               [7, 8, 9]])

A = np.array([[2, -1],
              [1,  3]])

print("\nA.T:")
print(A.T)
# Step 1: Compute A^TA
AtA = np.dot(A.T, A)

# Step 2: Find eigenvectors and eigenvalues of A^TA
eigenvalues, eigenvectors = np.linalg.eig(AtA)

# Perform SVD
U, Sigma, Vt = np.linalg.svd(A)

# Print the intermediate steps
print("Original Matrix A:")
print(A)

print("\nMatrix U0:")
print(U)

print("\nMatrix Sigma (Diagonal)0:")
print(np.diag(Sigma))

print("\nMatrix Vt (Transpose of V)0:")
print(Vt)
#U
# Example matrix A

# Step 1: Calculate AA^T
AAt = np.dot(A, A.T)

# Step 2: Find eigenvalues and eigenvectors of AA^T
eigenvalues, eigenvectors = np.linalg.eig(AtA)

# Step 3: Normalize eigenvectors
normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Matrix U is the normalized eigenvectors
U = normalized_eigenvectors

print("Matrix U:")
print(U)

# Sigma
# Step 1: Calculate A^TA
AtA = np.dot(A.T, A)

# Step 2: Find eigenvalues and eigenvectors of A^TA
eigenvalues, eigenvectors = np.linalg.eig(AtA)

# Step 3: Sort and extract singular values (square roots of eigenvalues)
singular_values = np.sqrt(np.sort(eigenvalues)[::-1])

# Step 4: Construct Sigma
Sigma = np.diag(singular_values)

print("Matrix Sigma:")
print(Sigma)

#Vt
# Example matrix A

# Step 1: Calculate A^TA
AtA = np.dot(A.T, A)

# Step 2: Find eigenvalues and eigenvectors of A^TA
eigenvalues, eigenvectors = np.linalg.eig(AtA)

# Step 3: Normalize eigenvectors
normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Step 4: Transpose V
Vt = normalized_eigenvectors.T

print("Matrix Vt:")
print(Vt)