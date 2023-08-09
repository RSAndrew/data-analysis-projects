import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium


# Generate dummy Airbnb data
np.random.seed(42)
num_samples = 1000

room_types = np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], size=num_samples)
neighborhoods = np.random.choice(['Downtown', 'Midtown', 'Uptown', 'Suburb'], size=num_samples)
number_of_reviews = np.random.randint(1, 200, size=num_samples)
prices = np.random.randint(50, 1000, size=num_samples)
latitude = np.random.uniform(37.7, 37.9, size=num_samples)
longitude = np.random.uniform(-122.5, -123, size=num_samples)

data = pd.DataFrame({
    'room_type': room_types,
    'neighborhood': neighborhoods,
    'number_of_reviews': number_of_reviews,
    'price': prices,
    'latitude': latitude,
    'longitude': longitude
})

# Save the dummy data to an Excel file
excel_file = 'airbnb_data.xlsx'
data.to_excel(excel_file, index=False)

# Load Airbnb data from an Excel file or CSV
excel_file = 'airbnb_data.xlsx'
data = pd.read_excel(excel_file)

# Display basic information about the dataset
print(data.info())

# Explore and visualize data
# Create bar plot of room types
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='room_type')
plt.title('Distribution of Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Create scatter plot of price vs. number of reviews
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='number_of_reviews', y='price', alpha=0.7)
plt.title('Price vs. Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Price')
plt.show()

# Create interactive map of listings using folium
map_airbnb = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)
for index, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue',
        popup=f"Price: ${row['price']}, Room Type: {row['room_type']}"
    ).add_to(map_airbnb)

# Save the map as an HTML file
map_airbnb.save('airbnb_map.html')

# Conduct geospatial analysis
popular_neighborhoods = data['neighborhood'].value_counts()[:10]

# Create bar plot of popular neighborhoods
plt.figure(figsize=(10, 6))
popular_neighborhoods.plot(kind='bar')
plt.title('Top 10 Popular Neighborhoods')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)
plt.show()
