#User-item Показує які послуги можна порекомендувати клієнту
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Replace this with your one-hot encoded DataFrame
data = pd.read_csv("./one_hot_encoded.csv")

one_hot_encoded_df = pd.DataFrame(data)
one_hot_encoded_df = one_hot_encoded_df.drop(columns="id")
one_hot_encoded_df = one_hot_encoded_df.drop(columns="Стать_клієнта")
# Set the 'Гість_' column as the index
one_hot_encoded_df.set_index('Гість_', inplace=True)

# Convert the one-hot encoded DataFrame to a sparse matrix
#Розріджена матриця (більша частина складається з нулів)
one_hot_encoded_sparse_matrix = csr_matrix(one_hot_encoded_df.values)

# Number of factors for matrix factorization
NUMBER_OF_FACTORS_MF = 15

# Perform matrix factorization using Truncated SVD
svd = TruncatedSVD(n_components=NUMBER_OF_FACTORS_MF)
user_factors = svd.fit_transform(one_hot_encoded_sparse_matrix)
item_factors = svd.components_

# Calculate the predicted ratings
all_user_predicted_ratings = np.dot(user_factors, item_factors)

# Convert the predicted ratings to a DataFrame
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns=one_hot_encoded_df.columns, index=one_hot_encoded_df.index)

# Example: Top recommendations for a specific user ("Балух Михайло")
user_to_recommend = "Hessenauer Robert"
top_recommendations = cf_preds_df.loc[user_to_recommend].sort_values(ascending=False)
print("Top Recommendations for User {}: \n{}".format(user_to_recommend, top_recommendations[0:20]))
