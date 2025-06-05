import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
# Replace this with your one-hot encoded DataFrame
# Load or create a DataFrame with user-item interactions
#whole dataset
data = pd.read_csv("first_dataset_summary.csv")
#load splited dataset first contain info from 2022-12-22 to 2023-08-20
train_data = pd.read_csv("first_dataset_summary.csv")
#second test dataset contain info from 2023-08-20 to 2023-12-21
test_data = pd.read_csv("second_dataset_summary.csv")
test_ = pd.DataFrame(test_data)
test_ = test_data.set_index('Гість_', inplace=True)
test_ = test_data.drop(columns=["id","Стать_клієнта"])

one_hot_encoded_df = pd.DataFrame(data)

# Set the 'Гість_' column as the index
one_hot_encoded_df.set_index('Гість_', inplace=True)
one_hot_encoded_df = one_hot_encoded_df.drop(columns=["id","Стать_клієнта"])

#func contain CF recommendation method
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

# Func contain item-based CF method
def content_based_recommendations(user, data, topn=10):
    # Drop irrelevant columns
    if 'id' in data.columns:
        data.drop(columns='id', inplace=True)
    if 'Стать_клієнта' in data.columns:
        data.drop(columns='Стать_клієнта', inplace=True)

    user_data = data.loc[user]

    # Transpose the DataFrame to have procedures as rows and users as columns
    transposed_data = data.transpose()

    # Compute cosine similarity between procedures
    item_similarity = cosine_similarity(transposed_data)

    # Get the indices of procedures sorted by similarity
    user_item_indices = user_data[user_data > 0].index
    recommended_items_indices = set()  # Use a set to store unique recommended item indices
    for item_name in user_item_indices:
        item_index = transposed_data.index.get_loc(item_name)
        similar_items_indices = item_similarity[item_index].argsort()[::-1][1:]  # Exclude the item itself
        recommended_items_indices.update(similar_items_indices)

    # Filter out items that the user has already interacted with
    recommended_items_indices -= set(user_item_indices)

    # Get the top recommended item names and their similarity scores
    recommended_items = [(transposed_data.index[i], item_similarity[item_index, i])
                         for i in recommended_items_indices]

    # Sort recommended items by similarity score in descending order
    recommended_items = sorted(recommended_items, key=lambda x: x[1], reverse=True)

    # Return the top recommended items
    return recommended_items[:topn]


def hybrid(user, data, topn=10, cb_ensemble_weight=0.7, cf_ensemble_weight=1.0):
    # Get recommendations from both collaborative filtering and content-based methods
    cb_recs = content_based_recommendations(user, data, topn)
    cf_recs = collaborative_filtering(user, data, topn)
    # Create DataFrames from the recommendations
    cf_recs_df = pd.DataFrame(cf_recs, columns=['item', 'recStrengthCF'])
    #print("CF",cf_recs_df)
    cb_recs_df = pd.DataFrame(cb_recs, columns=['item', 'recStrengthCB'])
    #print("CB",cb_recs_df)

    # Merge the recommendations based on the item column (procedure names)
    recs_df = cb_recs_df.merge(cf_recs_df, how='outer', left_on='item', right_on='item').fillna(0.0)
    #print(recs_df)
    # Compute the hybrid recommendation score based on CB and CF scores
    recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * cb_ensemble_weight) \
                                   + (recs_df['recStrengthCF'] * cf_ensemble_weight)
    #print(recs_df.shape)
    # Sort recommendations by the hybrid score in descending order
    recs_df = recs_df.sort_values(by='recStrengthHybrid', ascending=False).head(topn)

    # Return the top recommended items
    return recs_df[['item', 'recStrengthHybrid']]

# Calculate average precision to get recall
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

#Func of accuracy (recall, MAP, precision, f1k)
def recall_at_k(true_values, k=10, cb_ensemble_weight=0.7, cf_ensemble_weight=1.0):
    recall_values = []
    precision_values = []
    f1k_values = []
    ap_values = []
    i = 0
    for user in true_values.index:
        # Get indices of the top K recommended items for each user
        top_k_indices = hybrid(user,true_values,k,cb_ensemble_weight,cf_ensemble_weight).T.values[0]
        #print("TOP 10 RECOMMENDATIONS",top_k_indices,"END")
        i=i+1
        # Get true positive items for each user
        true_positives = true_values.loc[user][true_values.loc[user] > 0].index.tolist()
        #print("TRUE POSITIVE",true_positives)
        #print(set(true_positives))
        #print(set(top_k_indices) & set(true_positives))

        #Знаходимо суму поділених індексів однакових елементів
        ap_values.append(average_precision(top_k_indices, true_positives))
        # Calculate recall for the user
        recall_for_user = len(set(top_k_indices) & set(true_positives)) / float(len(true_positives))
        precision_for_user = len(set(top_k_indices) & set(true_positives)) / k
        #print(precision_for_user,recall_for_user,user)
        if (precision_for_user + recall_for_user)!=0:
            f1k_for_user = (2*recall_for_user*precision_for_user)/(precision_for_user+recall_for_user)
        else:
            f1k_for_user = 0
        #print(f1k_for_user)
        # Append recall value for the user
        recall_values.append(recall_for_user)
        precision_values.append(precision_for_user)
        f1k_values.append(f1k_for_user)
    # Calculate average recall across all users
    recall_at_k = np.mean(recall_values)
    precision_at_k = np.mean(precision_values)
    f1k_at_k= np.mean(f1k_values)
    map = (sum(ap_values)/len(true_values))
    return recall_at_k, precision_at_k,f1k_at_k,map

#Func to calculate accuracy with split dataset
def for_future(data, true_values, k=10, cb_ensemble_weight=0.5, cf_ensemble_weight=1.0):
    recall_values = []
    precision_values = []
    f1k_values = []
    ap_values = []
    i = 0
    for user in data.index:
        # Check if the user is present in true values DataFrame
        if user in true_values.index:
            # Get indices of the top K recommended items for each user
            top_k_indices = hybrid(user,data,k,cb_ensemble_weight,cf_ensemble_weight).T.values[0]
            print("\n", user)
            print("TOP 10 RECOMMENDATIONS", top_k_indices[1:5])
            i = i + 1
            # Get true positive items for each user
            true_positives = true_values.loc[user][true_values.loc[user] > 0].index.tolist()
            print("TRUE POSITIVE", true_positives)

            print(set(top_k_indices) & set(true_positives))

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

k=10
future = for_future(one_hot_encoded_df,test_,k)
print(f"Future (Prediction for splited df) hybrid method - \nRecall@{k}: {future[0]}\nPrecision@{k}: {future[1]}\nF1k@{k}: {future[2]}\nMAP@{k}: {future[3]}")
#user_name = "Корнієнко Ірина"
#recommendations = hybrid(user_name, one_hot_encoded_df,7)
#print("Top 10 Hybrid recommended procedures for user", user_name, ":", recommendations)
# cf = collaborative_filtering(user_name, one_hot_encoded_df,7)
# print("Top 10 CF recommended procedures for user", user_name, ":", cf)
# cb = content_based_recommendations(user_name, one_hot_encoded_df,7)
# print("Top 10 CB recommended procedures for user", user_name, ":", cb)
# recall_at_10 = recall_at_k(one_hot_encoded_df, k=10, cb_ensemble_weight=0.3, cf_ensemble_weight=1.0)
# print("Accurasy Recall@10: {}\nPrecision@10: {}\nF1k@10: {}\nMAP@10: {}".format(recall_at_10[0], recall_at_10[1],recall_at_10[2],recall_at_10[3]))