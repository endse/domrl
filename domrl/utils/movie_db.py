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
        
        # Initialize Embeddings
        self.embedding_dim = 16
        self.item_embeddings = None
        self.category_embeddings = np.random.randn(10, self.embedding_dim) # 10 Categories
        # Normalize
        self.category_embeddings /= np.linalg.norm(self.category_embeddings, axis=1, keepdims=True)
        self._build_item_embeddings()

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
        
    def _build_item_embeddings(self):
        """
        Constructs a simple embedding matrix for all movies.
        E(movie) = E(category) + small_random_noise
        This allows 'nearest neighbor' to find movies in the same genre, but distinct items.
        """
        # We need a dense index for movies (0 to N-1) to map to matrix rows
        # But movieId is sparse.
        self.all_movie_ids = self.movies['movieId'].values
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.all_movie_ids)}
        self.idx_to_movie_id = {i: mid for i, mid in enumerate(self.all_movie_ids)}
        
        num_movies = len(self.all_movie_ids)
        self.item_matrix = np.zeros((num_movies, self.embedding_dim), dtype=np.float32)
        
        for i, mid in enumerate(self.all_movie_ids):
            cat = self.movie_cat_map.get(mid, 0)
            base_emb = self.category_embeddings[cat]
            noise = np.random.normal(0, 0.1, size=self.embedding_dim)
            emb = base_emb + noise
            emb /= np.linalg.norm(emb) # Normalize
            self.item_matrix[i] = emb
            
    def search_nearest_items(self, query_emb, k=1):
        """
        Finds k nearest movies to the query embedding (dot product).
        query_emb: (Dim,)
        Returns: List of (movie_id, title, score)
        """
        # Dot product scores
        scores = np.dot(self.item_matrix, query_emb)
        
        # Top K
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            mid = self.idx_to_movie_id[idx]
            title = self.get_movie_title(mid)
            score = scores[idx]
            results.append((mid, title, score))
            
        return results

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
