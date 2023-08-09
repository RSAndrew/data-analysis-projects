import pandas as pd
import numpy as np

# Generate dummy data
num_samples = 300
np.random.seed(0)
age = np.random.randint(18, 65, num_samples)
income = np.random.randint(20000, 150000, num_samples)
spending_score = np.random.randint(1, 101, num_samples)

# Create a DataFrame
data = {
    'Age': age,
    'Income': income,
    'SpendingScore': spending_score
}
df = pd.DataFrame(data)

# Save the DataFrame to Excel
excel_file = 'customer_data.xlsx'
df.to_excel(excel_file, index=False)

# print('Excel file "customer_data.xlsx" with dummy data generated.')



# import pandas as pd
# import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load customer data from Excel
excel_file = 'customer_data.xlsx'
df = pd.read_excel(excel_file)

# Select relevant features for clustering
X = df[['Age', 'Income', 'SpendingScore']].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    plt.scatter(df[df['Cluster'] == cluster]['Income'], df[df['Cluster'] == cluster]['SpendingScore'], label=f'Cluster {cluster}')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.legend()
plt.show()