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


def collaborative_filtering(user, user_item_matrix, topn=5):
    if 'id' in user_item_matrix.columns:
        user_item_matrix.drop(columns='id', inplace=True)

    one_hot_encoded_sparse_matrix = csr_matrix(user_item_matrix.values, dtype=np.float64)
    NUMBER_OF_FACTORS_MF = 10

    # Performs matrix factorization using Truncated SVD
    U, sigma, Vt = svds(one_hot_encoded_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

    # Reconstruct the predicted ratings
    all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
            all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

    # Converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(
        all_user_predicted_ratings_norm,
        columns=user_item_matrix.columns,
        index=user_item_matrix.index,
    )

    # Get the top recommended items and their predicted ratings for the user
    top_recommendations = cf_preds_df.loc[user].sort_values(ascending=False).head(topn)

    # Convert the Series to a DataFrame with appropriate column names
    cf_recs_df = pd.DataFrame({'item': top_recommendations.index, 'recStrengthCF': top_recommendations.values})

    return cf_recs_df


def content_based_recommendations(user, data, topn=10):
    # Викидаємо непотрібні нам стовпці
    if 'id' in data.columns:
        data.drop(columns='id', inplace=True)
    if 'Стать_клієнта' in data.columns:
        data.drop(columns='Стать_клієнта', inplace=True)

    user_data = data.loc[user]

    # Транспонуємо наш датасет щоб мати клієнтів як стовпці а процедури як рядки
    transposed_data = data.transpose()

    # Знаходимо косинусну подібність між процедурами
    item_similarity = cosine_similarity(transposed_data)
    # Отримуємо індекси найподібніших процедур та сортуємо їх
    user_item_indices = user_data[user_data > 0].index
    recommended_items_indices = set()  # Use a set to store unique recommended item indices
    for item_name in user_item_indices:
        item_index = transposed_data.index.get_loc(item_name)
        similar_items_indices = item_similarity[item_index].argsort()[::-1][1:]  # Exclude the item itself
        recommended_items_indices.update(similar_items_indices)

    # Викидуємо ті процедури які клієнт вже пройшов
    recommended_items_indices -= set(user_item_indices)

    # Отримуємо топ процедур та їхню косинусну подібність
    recommended_items = [(transposed_data.index[i], item_similarity[item_index, i])
                         for i in recommended_items_indices]

    # Відсортовуємо процедури в спадному порядку
    recommended_items = sorted(recommended_items, key=lambda x: x[1], reverse=True)

    # Повертаємо топ найкращих послуг для клієнта
    return recommended_items[:topn]


def hybrid(user, data, topn=10, cb_ensemble_weight=0.7, cf_ensemble_weight=1.0):
    # Get recommendations from both collaborative filtering and content-based methods
    cb_recs = content_based_recommendations(user, data, topn)
    cf_recs = collaborative_filtering(user, data, topn)
    # Create DataFrames from the recommendations
    cf_recs_df = pd.DataFrame(cf_recs, columns=['item', 'recStrengthCF'])
    # print("CF",cf_recs_df)
    cb_recs_df = pd.DataFrame(cb_recs, columns=['item', 'recStrengthCB'])
    # print("CB",cb_recs_df)

    # Merge the recommendations based on the item column (procedure names)
    recs_df = cb_recs_df.merge(cf_recs_df, how='outer', left_on='item', right_on='item').fillna(0.0)
    # print(recs_df)
    # Compute the hybrid recommendation score based on CB and CF scores
    recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * cb_ensemble_weight) \
                                   + (recs_df['recStrengthCF'] * cf_ensemble_weight)
    # print(recs_df.shape)
    # Sort recommendations by the hybrid score in descending order
    recs_df = recs_df.sort_values(by='recStrengthHybrid', ascending=False).head(topn)

    # Return the top recommended items
    return recs_df[['item', 'recStrengthHybrid']]


# Calculate recall
def average_precision(recommendations, ground_truth):
    # Initialize variables
    num_relevant_items = 0
    precision_sum = 0.0

    # Iterate through recommendations
    for i, recommendation in enumerate(recommendations):
        if recommendation in ground_truth:
            num_relevant_items += 1
            precision_sum += num_relevant_items / (i + 1)  # Precision at position (i+1)

    # Calculate Average Precision
    if num_relevant_items == 0:
        return 0
    else:
        return precision_sum / num_relevant_items


# Другий тип рекол точності краший варіант
def recall_at_k(true_values, k=10, cb_ensemble_weight=0.7, cf_ensemble_weight=1.0):
    """
    Calculate Recall@K for the model predictions.

    Parameters:
    - predictions: DataFrame containing predicted ratings for each user-item pair
    - true_values: DataFrame containing true ratings (interactions) for each user-item pair
    - k: Number of top recommendations to consider

    Returns:
    - recall_at_k: Recall@K score
    """
    recall_values = []
    precision_values = []
    f1k_values = []
    ap_values = []
    i = 0
    for user in true_values.index:
        # Get indices of the top K recommended items for each user
        top_k_indices = hybrid(user, true_values, k, cb_ensemble_weight, cf_ensemble_weight).T.values[0]
        # print("TOP 10 RECOMMENDATIONS",top_k_indices,"END")
        i = i + 1
        # Get true positive items for each user
        true_positives = true_values.loc[user][true_values.loc[user] > 0].index.tolist()
        # print("TRUE POSITIVE",true_positives)
        # print(set(true_positives))
        # print(set(top_k_indices) & set(true_positives))

        # Знаходимо суму поділених індексів однакових елементів
        ap_values.append(average_precision(top_k_indices, true_positives))
        # Calculate recall for the user
        recall_for_user = len(set(top_k_indices) & set(true_positives)) / float(len(true_positives))
        precision_for_user = len(set(top_k_indices) & set(true_positives)) / k
        # print(precision_for_user,recall_for_user,user)
        if (precision_for_user + recall_for_user) != 0:
            f1k_for_user = (2 * recall_for_user * precision_for_user) / (precision_for_user + recall_for_user)
        else:
            f1k_for_user = 0
        # print(f1k_for_user)
        # Append recall value for the user
        recall_values.append(recall_for_user)
        precision_values.append(precision_for_user)
        f1k_values.append(f1k_for_user)
    # Calculate average recall across all users
    recall_at_k = np.mean(recall_values)
    precision_at_k = np.mean(precision_values)
    f1k_at_k = np.mean(f1k_values)
    map = (sum(ap_values) / len(true_values))
    return recall_at_k, precision_at_k, f1k_at_k, map


user_name = "Корнієнко Ірина"
recommendations = hybrid(user_name, one_hot_encoded_df, 7)
print("Top 7 Hybrid recommended procedures for user", user_name, ":", recommendations)
cf = collaborative_filtering(user_name, one_hot_encoded_df, 7)
print("Top 7 CF recommended procedures for user", user_name, ":", cf)
cb = content_based_recommendations(user_name, one_hot_encoded_df, 7)
print("Top 7 CB recommended procedures for user", user_name, ":", cb)
recall_at_10 = recall_at_k(one_hot_encoded_df, k=10, cb_ensemble_weight=0.5, cf_ensemble_weight=1.0)
print("Accurasy Recall@10: {}\nPrecision@10: {}\nF1k@10: {}\nMAP@10: {}".format(recall_at_10[0], recall_at_10[1],
                                                                                recall_at_10[2], recall_at_10[3]))