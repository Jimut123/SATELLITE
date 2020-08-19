import sqlite3
import numpy as np

conn = sqlite3.connect('map_db.sqlite')
cur = conn.cursor()
cur.executescript('''
CREATE TABLE IF NOT EXISTS data (
    min_lat_deg  REAL,
    max_lat_deg  REAL,
    min_lon_deg  REAL,
    max_lon_deg  REAL,
    visited INTEGER,
    key TEXT
);
''')

lapse = 0.05
for i in np.arange(22.35,23,lapse):
    for j in np.arange(88,88.5,lapse):
        cur.execute('''INSERT INTO data (min_lat_deg, max_lat_deg, min_lon_deg, max_lon_deg, visited, key) VALUES (?, ?, ?, ?, ?, ?) ''', (i,i+lapse,j,j+lapse,0,""))
conn.commit()


