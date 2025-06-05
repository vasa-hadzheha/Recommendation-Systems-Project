#Look for the "elbow" or point where the curve starts to flatten to determine an appropriate value for num.
# Adjust this based on your specific requirements and computational resources.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("one_hot_encoded.csv")

# Set the 'Гість_' column as the index
one_hot_encoded_df = pd.DataFrame(data)
one_hot_encoded_df.set_index('Гість_', inplace=True)
#Видаляємо айді та робимо стовпець Гість_ нашим айді індексом, також можна видалити Стать_клієнта бо воно впливає на точність
one_hot_encoded_df = one_hot_encoded_df.drop(columns="id")
one_hot_encoded_df = one_hot_encoded_df.drop(columns="Стать_клієнта")
# Assuming you have already computed singular values
singular_values = np.linalg.svd(one_hot_encoded_df, full_matrices=False, compute_uv=False)

# Plot the scree plot
plt.plot(singular_values)
plt.title("Scree Plot")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")
plt.show()