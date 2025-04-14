import psycopg2
import pandas as pd
import folium
from folium.plugins import HeatMap

conn = psycopg2.connect(
    dbname="taxi_data",
    user="owen",
    password="password123",
    host="localhost",
    port="5432"
)

query = """
SELECT pickup_latitude, pickup_longitude
FROM raw_trips
WHERE pickup_latitude IS NOT NULL AND pickup_longitude IS NOT NULL
  AND pickup_latitude BETWEEN 40.5 AND 41
  AND pickup_longitude BETWEEN -74.3 AND -73.5
LIMIT 10000;
"""

df = pd.read_sql(query, conn)
conn.close()

heat_data = df[['pickup_latitude', 'pickup_longitude']].values.tolist()

# Center of Manhattan
map_center = [40.75, -73.98]
m = folium.Map(location=map_center, zoom_start=12)

HeatMap(heat_data).add_to(m)

m.save("taxi_pickup_heatmap.html")
print("Heatmap saved to taxi_pickup_heatmap.html")
