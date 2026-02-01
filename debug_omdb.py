from domrl.utils.movie_db import get_movie_db
import pandas as pd
import os

def debug_omdb():
    dataset_path = "c:/Users/cy569/Downloads/ml-latest/dataset"
    print(f"Loading DB from {dataset_path}")
    db = get_movie_db(dataset_path)
    
    print(f"Movies: {len(db.movies)}")
    print(f"IMDb Map Size: {len(db.imdb_map)}")
    
    # Check simple ID (Toy Story = 1)
    # in links.csv: 1,0114709,862
    mid = 1
    imdb_id = db.imdb_map.get(mid)
    tmdb_id = db.tmdb_map.get(mid)
    
    print(f"Movie {mid}: IMDb={imdb_id} (Type: {type(imdb_id)}), TMDB={tmdb_id}")
    
    # Simulate App Logic
    if imdb_id:
        formatted_imdb = f"tt{int(imdb_id):07d}"
        print(f"Formatted IMDb: {formatted_imdb}")
        url = f"http://img.omdbapi.com/?i={formatted_imdb}&apikey=TESTKEY"
        print(f"Generated URL: {url}")
        
    # Check random sample
    print("Sampling...")
    items = db.sample_movies_with_id(0, 1)
    if items:
        mid, title = items[0]
        imdb = db.imdb_map.get(mid)
        print(f"Sample: {title} ({mid}) -> IMDb: {imdb}")

if __name__ == "__main__":
    debug_omdb()
