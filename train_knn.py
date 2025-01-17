import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
historical_data = pd.read_csv('data/historical_weather_data.csv')
observations = pd.read_csv('data/3_day_observations.csv')

# clean and preprocess the data
print("Cleaning and preprocessing 3-day observations...")
observations.rename(columns=lambda x: x.strip().replace(' ', '_').lower(), inplace=True)
observations['date'] = observations['date'].fillna(method='ffill')  # Fill missing dates
observations['precipitation'] = observations['precipitation'].map({'No': 0, 'Yes': 1})  # Convert to numeric
print("Cleaned 3-day observations:")
print(observations.head())

# Ensure precipitation in historical data is numeric
historical_data['precipitation'] = historical_data['precipitation'].map({'No': 0, 'Yes': 1})

# check and handle missing values
print(historical_data.isnull().sum())

# Handle missing values
historical_data.fillna({
    'temperature': historical_data['temperature'].mean(),
    'precipitation': 0,
    'cloud_cover': 'Unknown',  # If applicable, otherwise drop or handle differently
    'wind': 'calm'  # Default value
}, inplace=True)

print("Checking for missing values after handling:")
print(historical_data.isnull().sum())

# Step 4: Encode categorical features
print("Encoding categorical features...")
# Combine unique labels for encoding
cloud_cover_labels = pd.concat([
    historical_data['cloud_cover'], 
    observations['cloud_cover']
]).unique()

wind_labels = pd.concat([
    historical_data['wind'], 
    observations['wind']
]).unique()

# Fit LabelEncoder with combined labels
encoder = LabelEncoder()
encoder.fit(cloud_cover_labels)
observations['cloud_cover'] = encoder.transform(observations['cloud_cover'])
historical_data['cloud_cover'] = encoder.transform(historical_data['cloud_cover'])

encoder.fit(wind_labels)
observations['wind'] = encoder.transform(observations['wind'])
historical_data['wind'] = encoder.transform(historical_data['wind'])

# Step 5: Prepare historical features (X) and target (y)
print("Preparing features and target...")
X_historical = historical_data[['temperature', 'cloud_cover', 'wind', 'precipitation']]
y_historical = historical_data['temperature'].shift(-1).dropna()  # Target: Next day's temperature
X_historical = X_historical[:-1]  # Align X with y

# Step 6: Train-test split and scale data
print("Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X_historical, y_historical, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train the k-NN regressor
print("Training k-NN model...")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 8: Predict temperatures for 3-day observations
print("Predicting temperatures for 3-day observations...")
X_observations = observations[['temperature', 'cloud_cover', 'wind', 'precipitation']]
X_observations_scaled = scaler.transform(X_observations)

predicted_temperatures = knn.predict(X_observations_scaled)
observations['predicted_temperature'] = predicted_temperatures

# Step 9: Compare predictions with actual observations
print("Comparing predictions with actual observations...")
observations['temperature_diff'] = observations['predicted_temperature'] - observations['temperature']
print("Predictions vs. Actual:")
print(observations[['date', 'time_of_the_day', 'temperature', 'predicted_temperature', 'temperature_diff']])

# Save the comparison to a CSV file
observations.to_csv('3_day_observations_with_predictions.csv', index=False)
print("Comparison saved to '3_day_observations_with_predictions.csv'")

# Step 10: Visualize predictions vs. actual
plt.figure(figsize=(10, 6))
plt.plot(observations['temperature'], label="Actual Temperatures", marker='o')
plt.plot(observations['predicted_temperature'], label="Predicted Temperatures", marker='x')
plt.xlabel("Observation")
plt.ylabel("Temperature")
plt.legend()
plt.title("Actual vs. Predicted Temperatures for 3-Day Observations")
plt.show()

