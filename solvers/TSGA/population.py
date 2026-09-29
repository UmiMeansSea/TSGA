"""
population.py
=============
Population construction, evaluation, and heuristic seeding.

Contains:
    - build_initial_population
    - evaluate_population
    - get_heuristic_seeds

`get_heuristic_seeds` is the ONLY function in the TSGA package that
touches the approximators (MST / Christofides). If you swap out
heuristics, edit this function only.
"""
import os
import sys

from .operators import random_tour


# ---------------- Heuristic seeding ----------------

def _try_import_approximators():
    """Import MST / Christofides solvers. Returns (mst, chr) or (None, None)."""
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    try:
        from common.approximators.MST_solver    import solve_with_details as mst
        from common.approximators.christo_solver import solve_with_details as chr_
        return mst, chr_
    except Exception:
        return None, None


def get_heuristic_seeds(case_id, n, min_n_for_seeding=30):
    """
    Return a list of seed tours from MST and Christofides, or None.

    Returns None if:
        - instance is smaller than `min_n_for_seeding`
        - approximators cannot be imported
        - both heuristic solvers fail
    """
    if n < min_n_for_seeding:
        return None

    mst, chr_ = _try_import_approximators()
    if mst is None:
        return None

    seeds = []
    try:
        seeds.append(list(mst(case_id)["tour"][:-1]))    # drop closing dup
    except Exception:
        pass
    try:
        seeds.append(list(chr_(case_id)["tour"][:-1]))
    except Exception:
        pass

    return seeds or None


# ---------------- Population helpers ----------------

def build_initial_population(n, pop_size, rng, seeds=None):
    """
    Create `pop_size` tours. If `seeds` is provided, its tours
    overwrite the first slots (only those with correct length are used).
    """
    pop = [random_tour(n, rng) for _ in range(pop_size)]
    if seeds:
        for k, s in enumerate(seeds):
            if k < pop_size and len(s) == n:
                pop[k] = list(s)
    return pop


def evaluate_population(pop, fit_fn):
    """Return [fit_fn(t) for t in pop]."""
    return [fit_fn(t) for t in pop]