import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Завантажуємо датасет
data = pd.read_csv("one_hot_encoded.csv")

# Set the 'Гість_' column as the index
one_hot_encoded_df = pd.DataFrame(data)
one_hot_encoded_df.set_index('Гість_', inplace=True)
#Видаляємо айді та робимо стовпець Гість_ нашим айді індексрм, також можна видалити Стать_клієнта бо воно впливає на точність
one_hot_encoded_df = one_hot_encoded_df.drop(columns="id")
one_hot_encoded_df = one_hot_encoded_df.drop(columns="Стать_клієнта")

print(one_hot_encoded_df.columns)
# Convert the one-hot encoded DataFrame to a sparse matrix with a floating-point data type
#Перетворює матрицю з нульовими значеннями на атрицю де запам'ятовуються тільки не нульові значення та їхній індекс
#Використовується щоб зекономити пам'ять
one_hot_encoded_sparse_matrix = csr_matrix(one_hot_encoded_df.values, dtype=np.float64)
print(one_hot_encoded_sparse_matrix)
# The number of factors to factor the user-item matrix.
#Кількість факторів визначено методом ліктя
#Можна вказати кількість 15
NUMBER_OF_FACTORS_MF = 10
print('\n rank(A)=',np.linalg.matrix_rank(one_hot_encoded_df))

# Виконаємо матричну факторизацію SVD
U, sigma, Vt = svds(one_hot_encoded_sparse_matrix, k=NUMBER_OF_FACTORS_MF)

print ('A:\n', one_hot_encoded_sparse_matrix.toarray())
print ('=')
print ('\nU:\n', U)
print ('\nΣ:\n', sigma)
print('Matrix power (energy) =',np.sum(np.square(sigma)))
print ('\nV.T:\n', Vt)
# Перемножаємо матриці розкладу, щоб отримати матрицю реконструкції
all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
# Нормуємо значення щоб вони попадали в діапазон від 0 до 1
all_user_predicted_ratings_norm = (
    all_user_predicted_ratings - all_user_predicted_ratings.min()
) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

# Перетворюємо матрицю в датасет
cf_preds_df = pd.DataFrame(
    all_user_predicted_ratings_norm,
    columns=one_hot_encoded_df.columns,
    index=one_hot_encoded_df.index,
)
print("Реконструйована матриця \n",cf_preds_df)
# Function to get top recommendations for a specific user
def get_top_recommendations(user_to_recommend, cf_preds_df, topn=10):
    try:
        top_recommendations = cf_preds_df.loc[user_to_recommend].sort_values(ascending=False)
        return top_recommendations.head(topn)
    except KeyError:
        print("User '{}' not found in the recommendations.".format(user_to_recommend))
        return pd.Series([])

# Example: Top recommendations for a specific user
user_to_recommend = "Hessenauer Robert"
top_recommendations = get_top_recommendations(user_to_recommend, cf_preds_df)

# Display top recommendations
if not top_recommendations.empty:
    print("Top Recommendations for User '{}': \n{}".format(user_to_recommend, top_recommendations))
