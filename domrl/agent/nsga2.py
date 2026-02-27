"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II

Paper Section IV-D: Implements Pareto-optimal multi-objective
optimization for recommendation weight vectors.

Key innovations:
- Non-dominated sorting into fronts (F1, F2, ...)
- Crowding distance for diversity preservation
- Elitist selection strategy
- SBX crossover and polynomial mutation

Used as a 'pre-optimizer' for the weight space in the
Hybrid SAC-NSGA-II architecture (Paper Section IV-E).
"""

import numpy as np
from typing import List, Tuple


def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """
    Returns True if solution a dominates solution b.
    a dominates b if a is no worse in all objectives AND strictly better in at least one.
    
    Note: All objectives are assumed to be MAXIMIZED.
    For minimization objectives (e.g., churn), negate them before calling.
    """
    return np.all(obj_a >= obj_b) and np.any(obj_a > obj_b)


def non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Paper IV-D: Non-dominated sorting.
    Sorts population into Pareto fronts (F1, F2, ...).
    
    Args:
        objectives: (N, M) array where N=population size, M=num objectives
    
    Returns:
        List of fronts, each front is a list of indices
    """
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)  # How many solutions dominate i
    dominated_set = [[] for _ in range(n)]      # Solutions that i dominates
    
    fronts = [[]]
    
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
    
    # First front: solutions dominated by nobody
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Build subsequent fronts
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    
    # Remove last empty front
    if not fronts[-1]:
        fronts.pop()
    
    return fronts


def crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    """
    Paper IV-D: Crowding distance mechanism.
    
    Calculates the Manhattan distance between individuals and their
    closest neighbors in the objective space. This prevents the algorithm
    from concentrating on a single 'safe' approach, avoiding
    'algorithmic fatigue'.
    
    Args:
        objectives: (N, M) full objective matrix
        front: List of indices in this front
    
    Returns:
        distances: array of crowding distances for each member of the front
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    
    num_obj = objectives.shape[1]
    distances = np.zeros(n)
    
    front_objectives = objectives[front]
    
    for m in range(num_obj):
        # Sort by this objective
        sorted_indices = np.argsort(front_objectives[:, m])
        
        # Boundary points get infinite distance (always selected)
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Range for normalization
        obj_range = front_objectives[sorted_indices[-1], m] - front_objectives[sorted_indices[0], m]
        if obj_range < 1e-10:
            continue
        
        # Interior points
        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                front_objectives[sorted_indices[i + 1], m] - 
                front_objectives[sorted_indices[i - 1], m]
            ) / obj_range
    
    return distances


def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray, 
                  eta_c: float = 20.0, prob: float = 0.9,
                  lower: float = 0.0, upper: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) for real-valued vectors.
    
    Args:
        parent1, parent2: Parent weight vectors
        eta_c: Distribution index (higher = children closer to parents)
        prob: Crossover probability
        lower, upper: Bounds for the variables
    
    Returns:
        child1, child2: Offspring weight vectors
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    if np.random.rand() > prob:
        return child1, child2
    
    for i in range(len(parent1)):
        if np.random.rand() > 0.5:
            continue
            
        if abs(parent1[i] - parent2[i]) < 1e-10:
            continue
        
        # Calculate beta
        if parent1[i] < parent2[i]:
            y1, y2 = parent1[i], parent2[i]
        else:
            y1, y2 = parent2[i], parent1[i]
        
        rand = np.random.rand()
        
        # Beta calculation
        beta = 1.0 + (2.0 * (y1 - lower) / (y2 - y1 + 1e-10))
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        
        if rand <= (1.0 / alpha):
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        
        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        
        beta = 1.0 + (2.0 * (upper - y2) / (y2 - y1 + 1e-10))
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        
        if rand <= (1.0 / alpha):
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        
        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
        
        child1[i] = np.clip(c1, lower, upper)
        child2[i] = np.clip(c2, lower, upper)
    
    return child1, child2


def polynomial_mutation(individual: np.ndarray, eta_m: float = 20.0, 
                       prob: float = 0.1, lower: float = 0.0, 
                       upper: float = 5.0) -> np.ndarray:
    """
    Polynomial mutation for real-valued vectors.
    
    Args:
        individual: Weight vector to mutate
        eta_m: Distribution index (higher = smaller mutations)
        prob: Mutation probability per gene
        lower, upper: Bounds
    
    Returns:
        Mutated individual
    """
    mutant = individual.copy()
    
    for i in range(len(individual)):
        if np.random.rand() > prob:
            continue
        
        y = individual[i]
        delta1 = (y - lower) / (upper - lower + 1e-10)
        delta2 = (upper - y) / (upper - lower + 1e-10)
        
        rand = np.random.rand()
        mut_pow = 1.0 / (eta_m + 1.0)
        
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - val ** mut_pow
        
        y = y + deltaq * (upper - lower)
        mutant[i] = np.clip(y, lower, upper)
    
    return mutant


def compute_hypervolume_2d(objectives: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator for 2D objectives (approximation for higher dims).
    Used to track Pareto-front quality over training.
    
    For >2D, we compute a simple dominated hypervolume approximation by
    summing contributions of non-dominated points.
    """
    if len(objectives) == 0:
        return 0.0
    
    # Filter points that dominate the reference point (all worse)
    valid = np.all(objectives > ref_point, axis=1)
    if not np.any(valid):
        return 0.0
    
    filtered = objectives[valid]
    
    # Simple approximation: sum of rectangular contributions
    contributions = np.prod(filtered - ref_point, axis=1)
    return float(np.sum(contributions))


class NSGA2Optimizer:
    """
    Paper Section IV-D: NSGA-II for Multi-Objective Weight Optimization.
    
    Evolves a population of weight vectors to find Pareto-optimal
    trade-offs between Engagement, Satisfaction, Diversity, Fairness,
    and Churn Mitigation.
    
    Paper IV-D Table:
    - Standard NSGA-II: Crowding Distance & Non-dominated Sorting
    - Elitist approach: Best individuals preserved across generations
    """
    
    def __init__(self, num_objectives: int = 5, pop_size: int = 50,
                 num_generations: int = 20, crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1, eta_c: float = 20.0,
                 eta_m: float = 20.0, weight_lower: float = 0.0,
                 weight_upper: float = 5.0):
        self.num_objectives = num_objectives
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.weight_lower = weight_lower
        self.weight_upper = weight_upper
        
        # Initialize population randomly
        self.population = np.random.uniform(
            weight_lower, weight_upper, 
            size=(pop_size, num_objectives)
        )
        
        # Objective values for current population
        self.objectives = np.zeros((pop_size, num_objectives))
        
        # Best Pareto front
        self.pareto_front = []
        self.pareto_weights = np.array([])
        
    def set_objectives(self, objectives: np.ndarray):
        """
        Set the objective values for the current population.
        Called after evaluating each weight vector in the environment.
        """
        self.objectives = objectives.copy()
    
    def evaluate_weight_vector(self, weight_vector: np.ndarray, 
                                reward_vectors: List) -> np.ndarray:
        """
        Evaluate a weight vector by computing the per-objective 
        performance when using these weights.
        
        Args:
            weight_vector: (num_objectives,) weight vector
            reward_vectors: List of (num_objectives,) reward vectors from episodes
        
        Returns:
            (num_objectives,) averaged objective values
        """
        if len(reward_vectors) == 0:
            return np.zeros(self.num_objectives)
        
        reward_matrix = np.array(reward_vectors)
        return np.mean(reward_matrix, axis=0)
    
    def evolve(self) -> np.ndarray:
        """
        Paper IV-D: Run one generation of NSGA-II evolution.
        
        1. Non-dominated sorting
        2. Crowding distance assignment
        3. Selection, crossover, mutation
        4. Elitist combination
        
        Returns:
            Best weight vectors from the Pareto front
        """
        # --- Step 1: Create offspring ---
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Tournament selection (binary)
            i1, i2 = self._tournament_select(), self._tournament_select()
            
            # SBX Crossover
            child1, child2 = sbx_crossover(
                self.population[i1], self.population[i2],
                eta_c=self.eta_c, prob=self.crossover_prob,
                lower=self.weight_lower, upper=self.weight_upper
            )
            
            # Polynomial mutation
            child1 = polynomial_mutation(
                child1, eta_m=self.eta_m, prob=self.mutation_prob,
                lower=self.weight_lower, upper=self.weight_upper
            )
            child2 = polynomial_mutation(
                child2, eta_m=self.eta_m, prob=self.mutation_prob,
                lower=self.weight_lower, upper=self.weight_upper
            )
            
            offspring.append(child1)
            if len(offspring) < self.pop_size:
                offspring.append(child2)
        
        offspring = np.array(offspring[:self.pop_size])
        
        # --- Step 2: Combine parent + offspring (Elitist) ---
        combined_pop = np.vstack([self.population, offspring])
        combined_obj = np.vstack([self.objectives, self.objectives])  # Offspring use parent objectives initially
        
        # --- Step 3: Non-dominated sorting ---
        fronts = non_dominated_sort(combined_obj)
        
        # --- Step 4: Select next generation ---
        new_population = []
        new_objectives = []
        
        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                # Entire front fits
                for idx in front:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_obj[idx])
            else:
                # Partial front — sort by crowding distance
                distances = crowding_distance(combined_obj, front)
                sorted_by_crowd = np.argsort(-distances)  # Descending
                
                remaining = self.pop_size - len(new_population)
                for rank in sorted_by_crowd[:remaining]:
                    idx = front[rank]
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_obj[idx])
                break
        
        self.population = np.array(new_population)
        self.objectives = np.array(new_objectives)
        
        # --- Step 5: Extract Pareto front ---
        pareto_fronts = non_dominated_sort(self.objectives)
        if pareto_fronts:
            self.pareto_front = pareto_fronts[0]
            self.pareto_weights = self.population[self.pareto_front]
        
        return self.pareto_weights
    
    def _tournament_select(self) -> int:
        """Binary tournament selection based on domination rank."""
        i, j = np.random.randint(0, len(self.population), size=2)
        
        if dominates(self.objectives[i], self.objectives[j]):
            return i
        elif dominates(self.objectives[j], self.objectives[i]):
            return j
        else:
            return i if np.random.rand() < 0.5 else j
    
    def get_best_weights(self, strategy: str = "balanced") -> np.ndarray:
        """
        Get the best weight vector from the Pareto front.
        
        Args:
            strategy: 'balanced' (closest to ideal), 'engagement' (max engagement),
                      'trust' (max trust), 'random' (random Pareto member)
        
        Returns:
            (num_objectives,) best weight vector
        """
        if len(self.pareto_weights) == 0:
            return np.ones(self.num_objectives)
        
        if strategy == "random":
            idx = np.random.randint(0, len(self.pareto_weights))
            return self.pareto_weights[idx]
        
        elif strategy == "balanced":
            # Closest to the normalized ideal point
            pareto_obj = self.objectives[self.pareto_front]
            # Normalize objectives to [0, 1]
            mins = pareto_obj.min(axis=0)
            maxs = pareto_obj.max(axis=0)
            ranges = maxs - mins + 1e-10
            normalized = (pareto_obj - mins) / ranges
            
            # Ideal point is all 1s (max for all objectives)
            distances = np.linalg.norm(normalized - 1.0, axis=1)
            best_idx = self.pareto_front[np.argmin(distances)]
            return self.population[best_idx]
        
        elif strategy == "engagement":
            pareto_obj = self.objectives[self.pareto_front]
            best_idx = self.pareto_front[np.argmax(pareto_obj[:, 0])]
            return self.population[best_idx]
        
        elif strategy == "trust":
            pareto_obj = self.objectives[self.pareto_front]
            best_idx = self.pareto_front[np.argmax(pareto_obj[:, 1])]
            return self.population[best_idx]
        
        else:
            return self.pareto_weights[0]
    
    def get_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front for logging."""
        if len(self.pareto_front) == 0:
            return 0.0
        pareto_obj = self.objectives[self.pareto_front]
        ref_point = np.zeros(self.num_objectives)  # Origin as reference
        return compute_hypervolume_2d(pareto_obj, ref_point)
