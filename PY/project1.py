# create excel dummy file
import pandas as pd
import random

# Generate dummy customer data
num_customers = 100
customer_ids = range(1, num_customers + 1)
ages = [random.randint(18, 70) for _ in customer_ids]
incomes = [random.randint(20000, 150000) for _ in customer_ids]
spending_scores = [random.randint(1, 100) for _ in customer_ids]

# Create a DataFrame
data = {
    'CustomerID': customer_ids,
    'Age': ages,
    'Income': incomes,
    'SpendingScore': spending_scores
}
df = pd.DataFrame(data)

# Save DataFrame to Excel file
excel_file = 'customer_data.xlsx'
df.to_excel(excel_file, index=False)

# print(f'Dummy data saved to {excel_file}')



# import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load customer data from an Excel file
file_path = 'customer_data.xlsx'
df = pd.read_excel(file_path)

# Preprocess data: handle missing values and feature scaling
df.dropna(inplace=True)
features = df[['Age', 'Income', 'SpendingScore']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    silhouette_scores.append(silhouette_score(scaled_features, labels))

optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print("Optimal Number of Clusters:", optimal_num_clusters)

# Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

# Build a recommendation system based on user-item interactions
user_item_matrix = pd.pivot_table(df, values='SpendingScore', index='CustomerID', columns='ProductID')
user_item_matrix.fillna(0, inplace=True)

# Calculate user similarities using cosine similarity
user_similarities = user_item_matrix.corr(method='cosine')

# Get user's top n recommendations
def get_recommendations(user_id, n=5):
    user_scores = user_item_matrix.loc[user_id]
    similar_users = user_similarities[user_id].sort_values(ascending=False)
    similar_users = similar_users[similar_users.index != user_id]

    recommendations = []
    for similar_user_id, similarity_score in similar_users.iteritems():
        similar_user_scores = user_item_matrix.loc[similar_user_id]
        unrated_items = similar_user_scores[similar_user_scores.isnull()].index
        recommendations.extend(unrated_items)

        if len(recommendations) >= n:
            break

    return recommendations[:n]

# Example usage:
user_id = 123
recommended_items = get_recommendations(user_id)
print("Recommended Items for User", user_id, ":", recommended_items)