#!/usr/bin/env python3

import cv2
import numpy as np
import sqlite3
import sys
import os
from datetime import datetime

def create_rtabmap_db(png_path, output_db):
    """
    Convert a PNG map file to an RTAB-Map database.
    
    Args:
        png_path (str): Path to input PNG file
        output_db (str): Path for output database file
    """
    # Read and process the PNG file
    image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image file: {png_path}")
    
    # Convert to occupancy grid format
    # White (255) = free space, Black (0) = occupied
    # RTAB-Map uses: -1 = unknown, 0 = free, 100 = occupied
    grid = np.zeros(image.shape, dtype=np.int8)
    grid[image == 255] = 0      # Free space
    grid[image == 0] = 100      # Occupied
    grid[image == 127] = -1     # Unknown (if grey exists in image)

    # Create new database
    if os.path.exists(output_db):
        os.remove(output_db)
    
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    # Create required tables
    cursor.execute('''CREATE TABLE Info (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')

    cursor.execute('''CREATE TABLE Data (
        id INTEGER PRIMARY KEY,
        mapId INTEGER,
        weight REAL,
        stamp REAL,
        label TEXT,
        ground_truth_pose BLOB,
        velocity BLOB,
        gps BLOB,
        env_sensors BLOB
    )''')

    cursor.execute('''CREATE TABLE Node (
        id INTEGER PRIMARY KEY,
        map_id INTEGER,
        weight REAL,
        stamp REAL,
        label TEXT,
        ground_truth_pose BLOB,
        velocity BLOB,
        gps BLOB,
        env_sensors BLOB
    )''')

    # Insert map metadata
    current_time = datetime.now().timestamp()
    resolution = 0.03  # 5cm per pixel, adjust as needed
    
    info_data = [
        ('DatabaseVersion', '0.20.0'),
        ('GridCellSize', str(resolution)),
        ('created', str(current_time))
    ]
    cursor.executemany('INSERT INTO Info VALUES (?,?)', info_data)

    # Compress and store the occupancy grid
    grid_compressed = cv2.imencode('.png', grid)[1].tobytes()
    
    # Insert basic node with the grid
    cursor.execute('''INSERT INTO Node 
        (id, map_id, weight, stamp, label) 
        VALUES (1, 0, 1.0, ?, ?)''', 
        (current_time, grid_compressed))

    conn.commit()
    conn.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 png_to_rtabmap.py <input_png> <output_db>")
        sys.exit(1)
        
    input_png = sys.argv[1]
    output_db = sys.argv[2]
    
    try:
        create_rtabmap_db(input_png, output_db)
        print(f"Successfully converted {input_png} to {output_db}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()