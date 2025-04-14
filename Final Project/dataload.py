import psycopg2
import pandas as pd
from psycopg2.extras import execute_batch

print("Loading dataset...")
data = pd.read_csv("data/trip_data.csv")
print("Dataset loaded successfully!")

conn = psycopg2.connect(
    dbname="taxi_data",
    user="owen",
    password="password123",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS raw_trips (
    id TEXT PRIMARY KEY,
    vendor_id INTEGER,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count INTEGER,
    pickup_longitude FLOAT,
    pickup_latitude FLOAT,
    dropoff_longitude FLOAT,
    dropoff_latitude FLOAT,
    store_and_fwd_flag TEXT,
    trip_duration INTEGER
);
""")
conn.commit()
print("Table created successfully!")

rows = [
    (
        row['id'],
        int(row['vendor_id']),
        row['pickup_datetime'],
        row['dropoff_datetime'],
        int(row['passenger_count']),
        float(row['pickup_longitude']),
        float(row['pickup_latitude']),
        float(row['dropoff_longitude']),
        float(row['dropoff_latitude']),
        row['store_and_fwd_flag'],
        int(row['trip_duration'])
    )
    for _, row in data.iterrows()
]

query = """
    INSERT INTO raw_trips (
        id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count,
        pickup_longitude, pickup_latitude,
        dropoff_longitude, dropoff_latitude,
        store_and_fwd_flag, trip_duration
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING;
"""

print("Inserting data in batches...")
execute_batch(cur, query, rows, page_size=1000)
conn.commit()

cur.close()
conn.close()
print("Data inserted successfully.")
