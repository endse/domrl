from domrl.utils.movie_db import get_movie_db
import pandas as pd
import os

dataset_path = "c:/Users/cy569/Downloads/ml-latest/dataset"

def debug_links():
    print(f"Checking dataset at: {dataset_path}")
    links_file = os.path.join(dataset_path, "links.csv")
    if os.path.exists(links_file):
        print(f"links.csv found. Size: {os.path.getsize(links_file)} bytes")
    else:
        print("ERROR: links.csv NOT found!")
        return

    print("Initializing MovieDatabase...")
    db = get_movie_db(dataset_path)
    
    # Check map size
    map_size = len(db.tmdb_map)
    print(f"TMDB Map Size: {map_size}")
    
    if map_size == 0:
        print("ERROR: TMDB Map is empty! Check parsing logic.")
    else:
        # Check Toy Story (ID 1)
        tmdb_id = db.tmdb_map.get(1)
        print(f"Movie ID 1 (Toy Story) TMDB ID: {tmdb_id}")
        
    # Check sample
    print("Sampling a movie...")
    items = db.sample_movies_with_id(0, 1) # Action
    mid, title = items[0]
    tmdb = db.tmdb_map.get(mid)
    print(f"Sample: {title} (ID: {mid}) -> TMDB: {tmdb}")

if __name__ == "__main__":
    debug_links()
