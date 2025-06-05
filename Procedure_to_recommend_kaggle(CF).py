#Показує клієнта якому найбільше підійде дана пропозиція(або ж певна кореляція між клієнтами)
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Load or create a DataFrame with user-item interactions (replace 'one_hot_encoded.csv' with your file)
data = pd.read_csv("one_hot_encoded.csv")

# Set the 'Гість_' column as the index
one_hot_encoded_df = pd.DataFrame(data)
one_hot_encoded_df.set_index('Гість_', inplace=True)

# Convert the one-hot encoded DataFrame to a sparse matrix with a floating-point data type
one_hot_encoded_sparse_matrix = csr_matrix(one_hot_encoded_df.values, dtype=np.float64)

# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 10

# Performs matrix factorization using Truncated SVD
U, sigma, Vt = svds(one_hot_encoded_sparse_matrix, k=NUMBER_OF_FACTORS_MF)
U.shape, sigma.shape, Vt.shape
print(U.shape, sigma.shape, Vt.shape)
# Reconstruct the predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
all_user_predicted_ratings_norm = (
    all_user_predicted_ratings - all_user_predicted_ratings.min()
) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

# Converting the reconstructed matrix back to a Pandas dataframe
#Транспонуємо отримані значення для того щоб бачити клієнтів за запропонованою послугою
cf_preds_df = pd.DataFrame(
    all_user_predicted_ratings_norm,
    columns=one_hot_encoded_df.columns,
    index=one_hot_encoded_df.index,
).transpose()
print(cf_preds_df.head(10))
# Example: Top recommendations for a specific user
procedure_to_recommend = "Relaxation_massage"

try:
    top_recommendations = cf_preds_df.loc[procedure_to_recommend].sort_values(ascending=False)
    print("Top Recommendations for User {}: \n{}".format(procedure_to_recommend, top_recommendations[0:10]))
except KeyError:
    print("User '{}' not found in the recommendations.".format(procedure_to_recommend))
