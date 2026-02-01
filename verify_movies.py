from domrl.utils.movie_db import get_movie_db
import os

dataset_path = "c:/Users/cy569/Downloads/ml-latest/dataset"

def test_movie_db():
    print("Initializing MovieDatabase...")
    db = get_movie_db(dataset_path)
    
    print("\nTesting Category Mapping:")
    # Test a few categories
    cats = [0, 1, 5] # Action, Comedy, Horror
    for c in cats:
        movies = db.sample_movies_by_category(c, n=3)
        print(f"Category {c} samples: {movies}")
        assert len(movies) == 3, "Should return 3 movies"
        assert movies[0] != "Generic Title", "Should return real titles"
        
    print("\nTesting Specific Movie Title:")
    # Toy Story is usually ID 1
    title = db.get_movie_title(1)
    print(f"Movie ID 1: {title}")
    assert "Toy Story" in title, "Movie ID 1 should be Toy Story"

    print("\nVerification Passed!")

if __name__ == "__main__":
    if os.path.exists(dataset_path):
        test_movie_db()
    else:
        print(f"Dataset not found at {dataset_path}. Skipping test.")
