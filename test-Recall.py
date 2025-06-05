import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# Load or create a DataFrame with user-item interactions (replace 'one_hot_encoded.csv' with your file)
data = pd.read_csv("one_hot_encoded.csv")

# Set the 'Гість_' column as the index
one_hot_encoded_df = pd.DataFrame(data)
one_hot_encoded_df.set_index('Гість_', inplace=True)
# Видаляємо айді та робимо стовпець Гість_ нашим айді індексрм, також можна видалити Стать_клієнта бо воно впливає на точність
one_hot_encoded_df = one_hot_encoded_df.drop(columns="id")
one_hot_encoded_df = one_hot_encoded_df.drop(columns="Стать_клієнта")
#print(one_hot_encoded_df.columns)
# Convert the one-hot encoded DataFrame to a sparse matrix with a floating-point data type
# Перетворює матрицю з нульовими значеннями на матрицю, де запам'ятовуються тільки не нульові значення та їхній індекс.
# Використовується щоб зекономити пам'ять
one_hot_encoded_sparse_matrix = csr_matrix(one_hot_encoded_df.values, dtype=np.float64)
#print(one_hot_encoded_sparse_matrix)
# The number of factors to factor the user-item matrix.
# Кількість факторів визначено методом ліктя
# Можна вказати кількість 15
NUMBER_OF_FACTORS_MF = 10

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

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
#print(cf_preds_df)

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
        #print(precision_sum / num_relevant_items)
        return precision_sum / num_relevant_items

#Другий тип рекол точності краший варіант
def recall_at_k(predictions, true_values, k=10):
    #Якщо видалити Стать клієнта із дослідження точності то точність падає не сильно на 2%+-
    #predictions=predictions.drop(columns="Стать_клієнта")
    #true_values=true_values.drop(columns="Стать_клієнта")
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
    for user in predictions.index:
        # Get indices of the top K recommended items for each user
        top_k_indices = predictions.iloc[i].nlargest(k).to_frame().T
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

# Assuming you have a DataFrame 'true_values_df' containing true interactions
# For example, you might split your data into training and test sets, and 'true_values_df' is your test set.
# Make sure it has the same structure as 'one_hot_encoded_df'

# Calculate Recall@K for your collaborative filtering model
recall_at_5 = recall_at_k(cf_preds_df, one_hot_encoded_df, k=5)
recall_at_10 = recall_at_k(cf_preds_df, one_hot_encoded_df, k=10)
recall_at_15 = recall_at_k(cf_preds_df, one_hot_encoded_df, k=15)


print("Recall@15:", recall_at_15)
print("Accurasy Recall@10: {}\nPrecision@10: {}\nF1k@10: {}\nMAP@10: {}".format(recall_at_10[0], recall_at_10[1],recall_at_10[2],recall_at_10[3]))
print("Accurasy Recall@5: {}\nPrecision@5: {}\nF1k@5: {}\nMAP@5: {}".format(recall_at_5[0], recall_at_5[1],recall_at_5[2],recall_at_5[3]))