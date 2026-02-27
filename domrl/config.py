import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if it exists
start_path = Path(__file__).resolve().parent.parent
dotenv_path = start_path / '.env'
load_dotenv(dotenv_path)

class Config:
    # Dataset Paths
    MOVIE_LENS_PATH = os.getenv("MOVIE_LENS_PATH", str(Path.home() / "Downloads/ml-latest/dataset"))
    
    # API Keys
    OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")
    
    # Model Paths
    MODEL_CHECKPOINT_DIR = os.getenv("MODEL_CHECKPOINT_DIR", "logs")
    
    # Simulation Settings
    SLATE_SIZE = int(os.getenv("SLATE_SIZE", 3))
    HISTORY_LEN = int(os.getenv("HISTORY_LEN", 10))
    NUM_CATEGORIES = int(os.getenv("NUM_CATEGORIES", 10))
    
    # --- Paper Section III-A: Micro-Behavioral Thresholds ---
    SKIP_SHORT_THRESHOLD = float(os.getenv("SKIP_SHORT_THRESHOLD", 3.0))   # seconds
    SKIP_LONG_THRESHOLD = float(os.getenv("SKIP_LONG_THRESHOLD", 12.0))    # seconds
    
    # --- Paper Section III (Challenge C): Cold Start ---
    COLD_START_STEPS = int(os.getenv("COLD_START_STEPS", 5))  # Steps before persona inference
    
    # --- Paper Section IV-D: NSGA-II Configuration ---
    NSGA2_POP_SIZE = int(os.getenv("NSGA2_POP_SIZE", 50))         # Population size
    NSGA2_GENERATIONS = int(os.getenv("NSGA2_GENERATIONS", 20))    # Generations per evolution
    NSGA2_EVOLVE_INTERVAL = int(os.getenv("NSGA2_EVOLVE_INTERVAL", 10))  # Episodes between evolutions
    NSGA2_CROSSOVER_PROB = float(os.getenv("NSGA2_CROSSOVER_PROB", 0.9))
    NSGA2_MUTATION_PROB = float(os.getenv("NSGA2_MUTATION_PROB", 0.1))
    NSGA2_ETA_C = float(os.getenv("NSGA2_ETA_C", 20.0))  # SBX crossover distribution index
    NSGA2_ETA_M = float(os.getenv("NSGA2_ETA_M", 20.0))  # Polynomial mutation distribution index
    
    # --- Multi-Objective Dimensions ---
    NUM_OBJECTIVES = int(os.getenv("NUM_OBJECTIVES", 5))  # Engagement, Satisfaction, Diversity, Fairness, Churn
    
    # Debug
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

cfg = Config()
