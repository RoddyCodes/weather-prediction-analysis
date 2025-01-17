import pandas as pd

# Load your data
observations = pd.read_csv('data/3_day_observations.csv')
historical = pd.read_csv('data/historical_weather_data.csv')

# Combine datasets
combined = pd.concat([historical, observations], ignore_index=True)

# Drop duplicates and clean missing values
combined.dropna(inplace=True)

# Save combined data
combined.to_csv('combined_weather_data.csv', index=False)
print("Combined dataset saved to 'combined_weather_data.csv'")
