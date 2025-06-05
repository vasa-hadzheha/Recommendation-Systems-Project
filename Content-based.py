import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the DataFrame
data = pd.read_csv("./one_hot_encoded.csv")

# Set the 'Гість_' column as the index
data.set_index('Гість_', inplace=True)

# Drop irrelevant columns
data.drop(columns=["id", "Стать_клієнта"], inplace=True)

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

#Другий тип рекол точності краший варіант
def recall_at_k(true_values, k=10):
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
        top_k_indices = recommend_procedures(user,true_values,k)
        #print("TOP 10 RECOMMENDATIONS",top_k_indices)
        i=i+1
        # Get true positive items for each user
        true_positives = true_values.loc[user][true_values.loc[user] > 0].index.tolist()
        #print("TRUE POSITIVE",true_positives)

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

def recommend_procedures(user_name, data, num_recommendations=10):
    # Get the data for the specified user
    user_data = data.loc[user_name]

    # Transpose the DataFrame to have procedures as rows and users as columns
    transposed_data = data.transpose()

    # Compute cosine similarity between procedures
    item_similarity = cosine_similarity(transposed_data)

    # Get the indices of procedures sorted by similarity
    user_item_indices = user_data[user_data > 0].index
    recommended_items_indices = []
    for item_name in user_item_indices:
        try:
            item_index = transposed_data.index.get_loc(item_name)
            similar_items_indices = item_similarity[item_index].argsort()[::-1][1:]  # Exclude the item itself
            recommended_items_indices.extend(similar_items_indices)
        except KeyError:
            print(f"Item {item_name} not found, skipping...")

    # Filter out items that the user has already interacted with
    recommended_items_indices = [index for index in recommended_items_indices if index not in user_item_indices]

    # Get the top recommended item names
    recommended_items = transposed_data.iloc[recommended_items_indices].index.tolist()

    # Return the top recommended items
    return recommended_items[:num_recommendations]

# Example usage:
user_name = "Корнієнко Ірина"
recommendations = recommend_procedures(user_name, data)
print("Top 10 recommended procedures for user", user_name, ":", recommendations)
recall_at_10 = recall_at_k(data)
print(recall_at_10)
