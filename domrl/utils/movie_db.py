import pandas as pd
import numpy as np
import os

class MovieDatabase:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.movies = None
        self.links = None
        self.links = None
        self.tmdb_map = {} # movieId -> tmdbId
        self.imdb_map = {} # movieId -> imdbId (int)
        
        self.category_map = self._get_initial_category_map()
        self.movie_cat_map = {}
        self.cat_movie_map = {} # Reverse map: category -> [movie_ids]
        
        self.load_data()
        
    def _get_initial_category_map(self):
        return {
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

    def load_data(self):
        movies_file = os.path.join(self.dataset_path, "movies.csv")
        links_file = os.path.join(self.dataset_path, "links.csv")
        
        if not os.path.exists(movies_file):
            print(f"Error: {movies_file} not found.")
            return

        self.movies = pd.read_csv(movies_file)
        
        if os.path.exists(links_file):
            self.links = pd.read_csv(links_file)
            # Create mapping: movieId -> tmdbId
            # Filter out NaNs
            valid_links = self.links.dropna(subset=['tmdbId', 'imdbId'])
            self.tmdb_map = dict(zip(valid_links['movieId'], valid_links['tmdbId'].astype(int)))
            
            # Helper to safely parse IDs that might be strings with leading zeros
            def safe_int(x):
                try: return int(str(x))
                except: return 0
                
            self.imdb_map = dict(zip(valid_links['movieId'], valid_links['imdbId'].apply(safe_int)))
            
        
        # Precompute mappings
        self._build_mappings()
        
    def _get_category(self, genre_str):
        if not isinstance(genre_str, str): return 0
        first_genre = genre_str.split('|')[0]
        return self.category_map.get(first_genre, 0)

    def _build_mappings(self):
        self.movies['category'] = self.movies['genres'].apply(self._get_category)
        
        # 1. Movie -> Category
        self.movie_cat_map = dict(zip(self.movies['movieId'], self.movies['category']))
        
        # 2. Category -> List of Movies
        grouped = self.movies.groupby('category')
        self.cat_movie_map = {cat: group['movieId'].tolist() for cat, group in grouped}
        
    def get_movie_title(self, movie_id):
        if self.movies is None: return "Unknown Movie"
        row = self.movies[self.movies['movieId'] == movie_id]
        if not row.empty:
            return row.iloc[0]['title']
        return "Unknown Movie"
    
    def get_movie_poster(self, movie_id, api_key=None):
        """
        Returns a poster URL. 
        If api_key is provided, tries to fetch from TMDB.
        Else, returns a placeholder.
        """
        title = self.get_movie_title(movie_id)
        tmdb_id = self.tmdb_map.get(movie_id)
        
        if api_key and tmdb_id:
            return f"https://api.themoviedb.org/3/movie/{tmdb_id}/images?api_key={api_key}" 
            # Note: The above returns JSON. The UI needs to fetch it. 
            # Use a helper in the UI to resolve the final image path to avoid async here?
            # actually, let's return the basic construct and let the UI/Cache resolve it.
            # No, that's messy. 
            # Let's return a special object or just the ID for the UI to handle.
            pass
            
        # Fallback Placeholder
        safe_title = title.replace(" ", "+")
        return f"https://placehold.co/400x600/101010/FFFFFF/png?text={safe_title}"
    
    def sample_movies_by_category(self, category_idx, n=1):
        """Returns n random movie titles for a given category index."""
        # Delegating to new method for consistency, but returning just titles
        items = self.sample_movies_with_id(category_idx, n)
        return [item[1] for item in items]
        
    def sample_movies_with_id(self, category_idx, n=1):
        """Returns n random (id, title) tuples for a given category index."""
        if category_idx not in self.cat_movie_map:
            return [(0, "Generic Title")] * n
            
        candidate_ids = self.cat_movie_map[category_idx]
        if not candidate_ids:
            return [(0, "Generic Title")] * n
            
        # Sample IDs
        chosen_ids = np.random.choice(candidate_ids, size=min(n, len(candidate_ids)), replace=False)
        
        # Retrieve Titles
        results = []
        for mid in chosen_ids:
            title = self.get_movie_title(mid)
            results.append((mid, title))
            
        return results

# Singleton instance placeholder
_db_instance = None

def get_movie_db(dataset_path=None):
    global _db_instance
    if _db_instance is None and dataset_path:
        _db_instance = MovieDatabase(dataset_path)
    return _db_instance
