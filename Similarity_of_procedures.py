#Показує на скільки та чи інша послуга близька, щоб продавати послуги разом( до прикладу що Classic_massage(60min) часто беруть із Hydromassage_bath)
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Replace this with your one-hot encoded DataFrame
data = pd.read_csv("./one_hot_encoded.csv")

one_hot_encoded_df = pd.DataFrame(data)

# Set the 'Гість_' column as the index
one_hot_encoded_df.set_index('Гість_', inplace=True)
#Видаляємо непорібні стовпці для feature matrix
one_hot_encoded_df.drop(columns='id', inplace=True)
one_hot_encoded_df.drop(columns='Стать_клієнта', inplace=True)
# Calculate item-item similarity using cosine similarity
#Транспонується матриця та береться популярність кожної процедури до клієнта окремо так ми й отримуємо нашу feature matrix для процедур
item_similarity = cosine_similarity(one_hot_encoded_df.T)  # Transpose to get item-item similarity
# Create a DataFrame for item-item similarity
item_similarity_df = pd.DataFrame(item_similarity, columns=one_hot_encoded_df.columns, index=one_hot_encoded_df.columns)
print(item_similarity_df.index.sort_values())
# Example: Top item recommendations for a specific item ("Classic_massage(60min)")
item_to_recommend = "Classic_massage(60min)"
similar_items = item_similarity_df[item_to_recommend].sort_values(ascending=False)
print("Top Similar Items for Item {}: \n{}".format(item_to_recommend, similar_items[0:20]))

#скорочений варіант написання коду
item_to_recommend = "Hydromassage_bath"
new_item_similarity = pd.DataFrame(cosine_similarity(one_hot_encoded_df.T), index=one_hot_encoded_df.columns, columns=one_hot_encoded_df.columns)
print("Скорочений варіант коду",new_item_similarity[item_to_recommend].sort_values(ascending=False)[0:20])



