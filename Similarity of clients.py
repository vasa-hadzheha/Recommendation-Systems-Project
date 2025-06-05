import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Assuming one_hot_encoded is your DataFrame
# Load your data (replace 'your_data.csv' with your actual file path)
# one_hot_encoded = pd.read_csv('your_data.csv')
data = pd.read_csv("./one_hot_encoded.csv")

one_hot_encoded = pd.DataFrame(data)
# Extract relevant columns for modeling
features = one_hot_encoded.drop(columns=['id', 'Гість_', 'Стать_клієнта'])

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Compute cosine similarity
cosine_sim = cosine_similarity(features_standardized, features_standardized)

# Convert the cosine similarity matrix into a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=one_hot_encoded['Гість_'], columns=one_hot_encoded['Гість_'])
print("cosinsim df", cosine_sim_df)
# Function to get top recommendations for procedures for a given guest
def get_top_procedure_recommendations(guest_name, top_n=10):
    # Get the cosine similarity scores for the given guest
    sim_scores = cosine_sim_df[guest_name]
    print("sim score", sim_scores)
    # Sort the guests based on similarity scores in descending order
    sim_scores = sim_scores.sort_values(ascending=False)

    # Filter out the guest itself (similarity score of 1)
    sim_scores = sim_scores.drop(guest_name)

    # Get the top N most similar guests
    top_similar_guests = sim_scores.head(top_n).index

    # Get the procedures for the top N most similar guests
    recommended_procedures = one_hot_encoded[one_hot_encoded['Гість_'].isin(top_similar_guests)]

    # Print the top recommended procedures
    print(f"Top {top_n} recommendations for {guest_name} based on procedures of the most similar guests:")
    print(recommended_procedures)

# Example: Get top 10 recommendations for procedures for a specific guest
guest_name = "Hessenauer Robert"
get_top_procedure_recommendations(guest_name, top_n=10)
