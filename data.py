import requests
import pandas as pd
from datetime import datetime, timedelta

#junk weather api account 
# email: vesoti2965@dfesc.com
# password: password123
# WeatherAPI details
API_KEY = '631498d4e73e41be924200057251601'
LOCATION = 'Egg Harbor Township, NJ'
BASE_URL = 'http://api.weatherapi.com/v1/history.json'

#get the date range for last 6 months // 180 days
end_date = datetime.now()
start_date = end_date -timedelta(days=180)

#generate the dates
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days + 1)]

#make new empty array for weather of past  6 months
historical_weather = []

for date in dates:
    response = requests.get(
        BASE_URL,
        params={
            'key': API_KEY,
            'q': LOCATION,
            'dt': date
        }
    )

    #check to make sure status is 200 / ok
    if response.status_code == 200:
        print("200 OK status")
        data = response.json()
        for hour_data in data['forecast']['forecastday'][0]['hour']:
            historical_weather.append({
                'date': date,
                'time': hour_data['time'].split(' ')[1],
                'temperature': hour_data['temp_c'],
                'cloud_cover': hour_data['condition']['text'],
                'wind': 'calm' if hour_data['wind_kph'] < 10 else 'breezy' if hour_data['wind_kph'] < 20 else 'windy',
                'precipitation': hour_data['precip_mm']
            })
    else:
        print(f"Not sucessful status")

#add to new data frame // csv file for analysis
historical_df = pd.DataFrame(historical_weather)
historical_df.to_csv('historical_weather_data.csv', index=False)
print("data saved to 'historical_weather_data.csv'")
    