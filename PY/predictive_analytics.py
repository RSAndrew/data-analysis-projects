import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate
import random

# Function to generate dummy customer data
def generate_dummy_data(num_customers=100):
    data = []
    for _ in range(num_customers):
        customer_id = random.randint(1000, 9999)
        age = random.randint(18, 65)
        income = random.randint(20000, 150000)
        spending_score = random.randint(1, 10)
        data.append((customer_id, age, income, spending_score))
    return data

# Generate dummy customer data and save to Excel file
dummy_data = generate_dummy_data()
df = pd.DataFrame(dummy_data, columns=['CustomerID', 'Age', 'Income', 'SpendingScore'])
excel_file = 'customer_data.xlsx'
df.to_excel(excel_file, index=False)

# Load customer data from the Excel file
df = pd.read_excel(excel_file)

# Preprocess data
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[['Age', 'Income', 'SpendingScore']] = scaler.fit_transform(df_scaled[['Age', 'Income', 'SpendingScore']])

# Apply K-means clustering to segment customers
num_clusters = 4
X = df_scaled[['Age', 'Income', 'SpendingScore']].values
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(X)

# Evaluate clustering using silhouette score
silhouette_avg = silhouette_score(X, df_scaled['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Build a recommendation system based on user-item interactions
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(df_scaled[['CustomerID', 'ProductID', 'Rating']], reader)
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Perform cross-validation for recommendation system
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
