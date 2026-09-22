# ============================================================
#  Objective.py — Contract + Fitness Evaluator
#  Location: common/Objective.py
#  Companions: common/approximator/mst_solver.py
#              common/problem/{small,medium,large}_cases.py
# ============================================================
"""
ALGORITHM CONTRACT
==================

Every algorithm that wants to be scored by this Objective must follow:

  INPUT
  -----
  matrix : np.ndarray of shape (n, n), dtype=int
           Symmetric distance matrix. matrix[i][j] = distance city i -> j.
           matrix[i][i] = 0.

  OUTPUT
  ------
  tour : list or np.ndarray of length n
         A permutation of [0, 1, ..., n-1].
         The tour is treated as a CLOSED cycle: the last city connects
         back to the first. Do NOT repeat the first city at the end.

EXAMPLE
-------
  from Objective import get_instance, fitness

  matrix = get_instance("S1")          # load once
  tour   = my_algorithm(matrix)        # returns list of n city indices
  f      = fitness("S1", tour, matrix) # penalized fitness (int)

FITNESS RULE (minimization)
---------------------------
  cost  = sum of edge lengths around the closed tour
  B     = 2 * (MST double-tree tour cost)     [the ceiling]
  f     = cost                if cost <= B
        = cost + 1000         if cost >  B
"""

import os
import sys
import numpy as np

# ---------------- Paths ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# MST solver lives in common/approximator/
from approximators.MST_solver import solve as _mst_solve

PROBLEM_DIR = os.path.join(HERE, "problem")

CASE_FOLDER = {
    "S": "tsp_small",
    "M": "tsp_medium",
    "L": "tsp_large",
}

ALL_CASES = ["S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"]

# ---------------- Tunables ----------------
PENALTY_WEIGHT = 1000   # W, applied when cost > 2 * MST_cost
CEILING_FACTOR = 2      # B = CEILING_FACTOR * MST_cost

# ---------------- Caches ----------------
_matrix_cache  = {}     # case_id -> np.ndarray
_ceiling_cache = {}     # case_id -> int

# ---------------- Instance loading ----------------
def get_instance(case_id):
    """
    Load and cache the distance matrix for a case ('S1'..'L3').

    Returns
    -------
    np.ndarray, shape (n, n), dtype=int
    """
    case_id = case_id.strip().upper()
    if case_id in _matrix_cache:
        return _matrix_cache[case_id]

    if not case_id or case_id[0] not in CASE_FOLDER:
        raise ValueError(f"Unknown case id: {case_id!r}. "
                         f"Expected one of {ALL_CASES}.")
    folder = CASE_FOLDER[case_id[0]]
    path   = os.path.join(PROBLEM_DIR, folder, f"{case_id}_matrix.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Matrix not found for {case_id}: {path}\n"
            f"Did you run the generator for {folder}?"
        )

    matrix = np.loadtxt(path, delimiter=",", dtype=int)
    _matrix_cache[case_id] = matrix
    return matrix

# ---------------- Ceiling ----------------
def get_ceiling(case_id):
    """
    B = CEILING_FACTOR * (MST double-tree tour cost).
    Cached per case — the MST solver runs at most once per case.
    """
    case_id = case_id.strip().upper()
    if case_id in _ceiling_cache:
        return _ceiling_cache[case_id]

    mst_cost = _mst_solve(case_id)
    ceiling  = CEILING_FACTOR * int(mst_cost)
    _ceiling_cache[case_id] = ceiling
    return ceiling

# ---------------- Core scoring ----------------
def tour_cost(tour, matrix):
    """Raw closed-tour length. Does NOT apply penalty."""
    n = len(matrix)
    if len(tour) != n:
        raise ValueError(f"Tour length {len(tour)} != matrix size {n}.")
    total = 0
    for i in range(n - 1):
        total += int(matrix[tour[i]][tour[i + 1]])
    total += int(matrix[tour[-1]][tour[0]])   # close the cycle
    return total

def _validate_tour(tour, n):
    if len(tour) != n:
        raise ValueError(f"Tour length {len(tour)} != n={n}.")
    if set(int(x) for x in tour) != set(range(n)):
        raise ValueError("Tour is not a permutation of 0..n-1.")

def fitness(case_id, tour, matrix=None):
    """
    Penalized fitness for a single tour. The algorithm calls this to
    know whether its solution is acceptable or above the ceiling.

    Parameters
    ----------
    case_id : str        one of 'S1'..'L3'
    tour    : sequence   permutation of 0..n-1 (closed cycle)
    matrix  : optional   pass the already-loaded matrix to skip I/O

    Returns
    -------
    int  ->  cost              if cost <= B
         ->  cost + PENALTY    if cost >  B
    """
    if matrix is None:
        matrix = get_instance(case_id)

    _validate_tour(tour, len(matrix))
    cost    = tour_cost(tour, matrix)
    ceiling = get_ceiling(case_id)

    if cost > ceiling:
        return cost + PENALTY_WEIGHT
    return cost

def evaluate(case_id, tour, matrix=None):
    """
    Full breakdown — for logging / debugging.
    Returns a dict with cost, ceiling, penalty, fitness, and a flag.
    """
    if matrix is None:
        matrix = get_instance(case_id)

    _validate_tour(tour, len(matrix))
    cost    = tour_cost(tour, matrix)
    ceiling = get_ceiling(case_id)
    penalty = PENALTY_WEIGHT if cost > ceiling else 0

    return {
        "case_id"  : case_id.upper(),
        "n"        : len(matrix),
        "cost"     : cost,
        "ceiling"  : ceiling,
        "penalty"  : penalty,
        "fitness"  : cost + penalty,
        "penalized": penalty > 0,
    }

# ---------------- Misc helpers ----------------
def cases():
    """Return the canonical list of all 9 case IDs."""
    return list(ALL_CASES)

def clear_cache():
    """Drop cached matrices and ceilings (useful between experiments)."""
    _matrix_cache.clear()
    _ceiling_cache.clear()

# ---------------- Smoke test ----------------
if __name__ == "__main__":
    # Sanity check: use the MST solver's own tour as a trivially valid input.
    from approximators.MST_solver import solve_with_details

    for cid in ALL_CASES:
        details = solve_with_details(cid)
        tour    = details["tour"]
        result  = evaluate(cid, tour)

        print(f"{cid:>3} | n={result['n']:>3} "
              f"| cost={result['cost']:>9} "
              f"| ceiling={result['ceiling']:>9} "
              f"| penalty={result['penalty']:>5} "
              f"| fitness={result['fitness']:>9} "
              f"| {'PENALIZED' if result['penalized'] else 'ok'}")