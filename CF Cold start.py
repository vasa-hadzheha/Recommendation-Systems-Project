import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


# Load or create a DataFrame with user-item interactions (replace 'one_hot_encoded.csv' with your file)
data = pd.read_csv("one_hot_encoded.csv")

# Set the 'Гість_' column as the index
one_hot_encoded_df = pd.DataFrame(data)
one_hot_encoded_df.set_index('Гість_', inplace=True)
one_hot_encoded_df = one_hot_encoded_df.drop(columns="id")

# Convert the one-hot encoded DataFrame to a sparse matrix with a floating-point data type
one_hot_encoded_sparse_matrix = csr_matrix(one_hot_encoded_df.values, dtype=np.float64)

# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15

# Performs matrix factorization using Truncated SVD
U, sigma, Vt = svds(one_hot_encoded_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

# Reconstruct the predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
all_user_predicted_ratings_norm = (
    all_user_predicted_ratings - all_user_predicted_ratings.min()
) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

# Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(
    all_user_predicted_ratings_norm,
    columns=one_hot_encoded_df.columns,
    index=one_hot_encoded_df.index,
)

# Function to get top recommendations for a specific user
def get_top_recommendations(user_to_recommend, cf_preds_df, topn=10):
    try:
        top_recommendations = cf_preds_df.loc[user_to_recommend].sort_values(ascending=False)
        return top_recommendations.head(topn)
    except KeyError:
        print("User '{}' not found in the recommendations.".format(user_to_recommend))
        return pd.Series([])

# Example: Top recommendations for a specific user
user_to_recommend = "Корнієнко Ірина"
top_recommendations = get_top_recommendations(user_to_recommend, cf_preds_df)

# Display top recommendations
if not top_recommendations.empty:
    print("Top Recommendations for User '{}': \n{}".format(user_to_recommend, top_recommendations))

def hybrid_recommendations(user_to_recommend, cf_preds_df, content_features_df, topn=20, alpha=0.5):
    try:
        # Get collaborative filtering recommendations
        cf_recommendations = get_top_recommendations(user_to_recommend, cf_preds_df, topn=topn)
        print("Collaborative Filtering Recommendations:\n", cf_recommendations)

        # Get content-based recommendations
        content_features = content_features_df.values
        print("\nContent Features Matrix:\n", content_features)

        similarity_scores = cosine_similarity([content_features_df.loc[user_to_recommend]], content_features)
        print("\nSimilarity Scores Matrix:\n", similarity_scores)
        print("\ncf_preds_df.values:\n", cf_preds_df.values)
        print("\nsimilarity_scores:\n", similarity_scores)

        content_based_scores = np.dot(similarity_scores, cf_preds_df.values)
        print("\nContent-Based Scores Matrix:\n", content_based_scores)

        content_based_recommendations = pd.Series(content_based_scores.flatten(), index=cf_preds_df.columns)
        content_based_recommendations = content_based_recommendations.sort_values(ascending=False).head(topn)
        print("\nContent-Based Recommendations:\n", content_based_recommendations)

        # Combine recommendations using a weighted average
        hybrid_recommendations = alpha * cf_recommendations + (1 - alpha) * content_based_recommendations
        print("\nHybrid Recommendations (Weighted Average):\n", hybrid_recommendations.sort_values(ascending=False))

        return hybrid_recommendations.sort_values(ascending=False).head(topn)
    except KeyError:
        print("User '{}' not found in the recommendations.".format(user_to_recommend))
        return pd.Series([])

# def hybrid_recommendations(user_to_recommend, cf_preds_df, content_features_df, topn=25, alpha=0.5):
#     try:
#         # Get collaborative filtering recommendations
#         cf_recommendations = get_top_recommendations(user_to_recommend, cf_preds_df, topn=topn)
#
#         # Get content-based recommendations
#         content_features = content_features_df.values  # Use all content features
#         similarity_scores = cosine_similarity([content_features_df.loc[user_to_recommend]], content_features)
#         content_based_scores = np.dot(similarity_scores, cf_preds_df.values)
#         content_based_recommendations = pd.Series(content_based_scores.flatten(), index=cf_preds_df.columns)
#         content_based_recommendations = content_based_recommendations.sort_values(ascending=False).head(topn)
#
#         # Combine recommendations using a weighted average
#         hybrid_recommendations = alpha * cf_recommendations + (1 - alpha) * content_based_recommendations
#
#         return hybrid_recommendations.head(topn)
#     except KeyError:
#         print("User '{}' not found in the recommendations.".format(user_to_recommend))
#         return pd.Series([])
#
#
# Example: Hybrid recommendations for a specific user
#user_to_recommend = "210_чоловік"
hybrid_top_recommendations = hybrid_recommendations(user_to_recommend, cf_preds_df, one_hot_encoded_df)

# Display hybrid recommendations
if not hybrid_top_recommendations.empty:
    print("Hybrid Recommendations for User '{}': \n{}".format(user_to_recommend, hybrid_top_recommendations))

# Example simple data
# user_to_recommend = "User1"
# cf_preds_df = pd.DataFrame({
#     "Item1": [0.2, 0.5],
#     "Item2": [0.8, 0.3],
# }, index=["User1", "User2"])
#
# content_features_df = pd.DataFrame({
#     "Feature1": [1.0, 0.5],
#     "Feature2": [0.2, 0.9],
# }, index=["User1", "User2"])
#
# # Call the hybrid_recommendations function
# hybrid_top_recommendations = hybrid_recommendations(user_to_recommend, cf_preds_df, content_features_df)
#
# # Display the hybrid recommendations
# print("\nFinal Hybrid Recommendations for User '{}': \n{}".format(user_to_recommend, hybrid_top_recommendations))