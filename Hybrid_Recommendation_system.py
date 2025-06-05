import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
# Replace this with your one-hot encoded DataFrame
data = pd.read_csv("./one_hot_encoded.csv")

one_hot_encoded_df = pd.DataFrame(data)

# Set the 'Гість_' column as the index
one_hot_encoded_df.set_index('Гість_', inplace=True)

def collaborative_filtering_with_cold_start(user, user_item_matrix, topn=10):
    # Check if the user is new
    if user not in user_item_matrix.index:
        # Handle cold start for new users using content-based recommendations
        recommendations = knowledge_based_recommendations(user_item_matrix,topn)
    else:
        # For existing users, use collaborative filtering
        recommendations = collaborative_filtering(user, user_item_matrix, topn)

    return recommendations

def knowledge_based_recommendations(one_hot_encoded_df, topn=10):
    one_hot_encoded_df.drop(columns='id', inplace=True)
    one_hot_encoded_df.drop(columns='Стать_клієнта', inplace=True)
    # Array of top most popular procedures in data set
    procedures = ["Classic_massage(60min)","Back_massage","Relaxation_massage","Shokewave_therapy","Oxy","Super_inductive_system",
                  "Hydromassage_bath","Classic_massage(90min)","Ultraphonophoresis","Aparat_limfostim(30min)",
                  "Sharko_shower","Magnetotherapy","Underwater_manual_massage","Lymphatic_drainage_massage","Dry_CO2","Sport_massage",
                  "Aparat_lymfostim(45min)","Individual_massage(45min)","Robolex_face","Pearl_bath_with_color_therapy","Stone_massage"]

    # Let the user choose one procedure from the list
    print("Choose one procedure from the list:")
    for i, procedure in enumerate(procedures, 1):
        print(f"{i}. {procedure}")

    # Get user input for the chosen procedure
    chosen_index = int(input("Enter the number corresponding to your chosen procedure: ")) - 1
    item_to_recommend = procedures[chosen_index]

    print("You have chosen:", item_to_recommend)

    new_item_similarity = pd.DataFrame(cosine_similarity(one_hot_encoded_df.T), index=one_hot_encoded_df.columns,
                                       columns=one_hot_encoded_df.columns)
    top_recommendations = new_item_similarity[item_to_recommend].sort_values(ascending=False)[0:topn]
    #print("Можемо порекомендувати вам", top_recommendations)
    print(topn)
    return top_recommendations

def collaborative_filtering(user, user_item_matrix, topn=5):
    user_item_matrix.drop(columns='id', inplace=True)
    one_hot_encoded_sparse_matrix = csr_matrix(user_item_matrix.values, dtype=np.float64)
    # The number of factors to factor the user-item matrix.
    # Кількість факторів визначено методом ліктя
    # Можна вказати кількість 15
    NUMBER_OF_FACTORS_MF = 10

    # Performs matrix factorization using Truncated SVD
    U, sigma, Vt = svds(one_hot_encoded_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

    # Reconstruct the predicted ratings
    all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

    # Converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(
        all_user_predicted_ratings_norm,
        columns=one_hot_encoded_df.columns,
        index=one_hot_encoded_df.index,
    )
    top_recommendations = cf_preds_df.loc[user].sort_values(ascending=False)
    return top_recommendations.head(topn)

# Example: Get recommendations for a new user
new_user_recommendations = collaborative_filtering_with_cold_start("Олена",one_hot_encoded_df)
print("Recommendations for User:\n", new_user_recommendations)