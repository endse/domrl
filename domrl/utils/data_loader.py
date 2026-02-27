import pandas as pd
import numpy as np
import os
from collections import deque

def load_movielens_data(dataset_path, history_len=10, max_rows=100000):
    """
    Loads MovieLens data and converts it into transitions for the DOM-RL environment.
    
    Args:
        dataset_path (str): Path to the directory containing ratings.csv and movies.csv.
        history_len (int): Length of the history window.
        max_rows (int): Limit rows for faster loading during dev.
        
    Returns:
        list: A list of transitions tuples (state, action, next_state, reward, done).
    """
    print(f"Loading MovieLens data from {dataset_path}...")
    
    ratings_file = os.path.join(dataset_path, "ratings.csv")
    movies_file = os.path.join(dataset_path, "movies.csv")
    
    # Load Data
    # ratings: userId, movieId, rating, timestamp
    ratings = pd.read_csv(ratings_file, nrows=max_rows)
    # movies: movieId, title, genres
    movies = pd.read_csv(movies_file)
    
    # Map Genres to Categories (0-9)
    # MovieLens Genres: Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy,
    # Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, (no genres listed)
    
    genre_map = {
        "Action": 0, "Adventure": 0, "War": 0, 
        "Comedy": 1, "Children": 1, "Animation": 1,
        "Drama": 2, "Romance": 2,
        "Sci-Fi": 3, "Fantasy": 3,
        "Crime": 4, "Mystery": 4, "Thriller": 4,
        "Horror": 5,
        "Documentary": 6,
        "Musical": 7,
        "Western": 8,
        "Film-Noir": 9,
        # Fallbacks
        "(no genres listed)": 0 
    }
    
    def get_category(genre_str):
        if not isinstance(genre_str, str): return 0
        first_genre = genre_str.split('|')[0]
        return genre_map.get(first_genre, 0) # Default to 0 if unknown
        
    # Create a movieId -> category map
    movies['category'] = movies['genres'].apply(get_category)
    movie_cat_map = dict(zip(movies['movieId'], movies['category']))
    
    # Sort ratings by User and Time
    ratings = ratings.sort_values(by=['userId', 'timestamp'])
    
    transitions = []
    
    grouped = ratings.groupby('userId')
    
    print(f"Processing {len(grouped)} users...")
    
    for user_id, user_data in grouped:
        # Initial empty history
        history = deque([0]*history_len, maxlen=history_len)
        
        # Convert user_data to list of dicts for faster iteration
        user_rows = user_data.to_dict('records')
        
        for i in range(len(user_rows)):
            row = user_rows[i]
            movie_id = row['movieId']
            rating = row['rating']
            
            action = movie_cat_map.get(movie_id, 0)
            
            # Construct State
            # Mocking user_features and micro_signals as they don't exist in MovieLens
            state = {
                "history": np.array(history, dtype=np.int32),
                "user_features": np.array([0.5, 0.0], dtype=np.float32), 
                "micro_signals": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "weights": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), # Default weights (Eng, Sat, Div, Fair)
                "persona_id": np.array([0], dtype=np.int32)
            }
            
            # Reward: Normalize 0.5-5.0 to 0.0-1.0
            # Simulating satisfaction
            reward_scalar = (rating - 0.5) / 4.5 
            
            # Next State
            next_history = history.copy()
            next_history.append(action)
            
            next_state = {
                "history": np.array(next_history, dtype=np.int32),
                "user_features": np.array([0.5, 0.0], dtype=np.float32), 
                "micro_signals": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "weights": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "persona_id": np.array([0], dtype=np.int32)
            }
            
            done = False
            # Mark end of user session as done (simplified)
            if i == len(user_rows) - 1:
                done = True
                
            transitions.append((state, action, next_state, reward_scalar, done))
            
            # Update history for next step
            history.append(action)
            
    print(f"Generated {len(transitions)} transitions.")
    return transitions

def load_user_sequences(dataset_path, min_len=5, max_rows=100000):
    """
    Loads MovieLens data and returns a list of action sequences (category indices) per user.
    Used for training the World Model (UserDynamicsNet).
    """
    import os
    import pandas as pd
    
    ratings_file = os.path.join(dataset_path, "ratings.csv")
    movies_file = os.path.join(dataset_path, "movies.csv")
    
    if not os.path.exists(ratings_file) or not os.path.exists(movies_file):
        raise FileNotFoundError("Dataset files not found.")
        
    # Load Movies & Map to Categories
    movies = pd.read_csv(movies_file)
    
    genre_map = {
        "Action": 0, "Adventure": 0, "War": 0, 
        "Comedy": 1, "Children": 1, "Animation": 1,
        "Drama": 2, "Romance": 2,
        "Sci-Fi": 3, "Fantasy": 3,
        "Crime": 4, "Mystery": 4, "Thriller": 4,
        "Horror": 5,
        "Documentary": 6,
        "Musical": 7,
        "Western": 8,
        "Film-Noir": 9,
        "(no genres listed)": 0 
    }
    
    def get_category(genre_str):
        if not isinstance(genre_str, str): return 0
        first_genre = genre_str.split('|')[0]
        return genre_map.get(first_genre, 0)

    movies['category'] = movies['genres'].apply(get_category)
    movie_cat_map = dict(zip(movies['movieId'], movies['category']))
    
    # Load Ratings
    ratings = pd.read_csv(ratings_file, nrows=max_rows)
    ratings = ratings.sort_values(by=['userId', 'timestamp'])
    
    sequences = []
    
    grouped = ratings.groupby('userId')
    print(f"Extracting sequences from {len(grouped)} users...")
    
    for user_id, user_data in grouped:
        seq = []
        for movie_id in user_data['movieId']:
            cat = movie_cat_map.get(movie_id, 0)
            seq.append(cat)
        
        if len(seq) >= min_len:
            sequences.append(seq)
            
    print(f"Extracted {len(sequences)} valid sequences.")
    return sequences
