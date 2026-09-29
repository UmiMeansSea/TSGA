"""
main.py
=======
Public API for TSGA.

This is the ONLY file experiment.py should import:

    from solvers.TSGA.main import solve, solve_with_details, run_on_cases

Binds the internal TSGA pipeline (params → population → ga_engine)
to the shared TSP objective (common/objectives.py).

To change the public interface (return shape, extra metadata):
edit only this file.
"""
import os
import sys
import time
import numpy as np

# ---------------- Path setup ----------------
HERE = os.path.dirname(os.path.abspath(__file__))            # TSGA/solvers/TSGA
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))       # TSGA
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# From common/
from common.objectives import (
    fitness,
    get_instance,
    get_ceiling,
    tour_cost,
    ALL_CASES,
)

# From this package
from .params     import default_params
from .ga_engine  import ga_loop
from .population import get_heuristic_seeds


# ---------------- Core driver ----------------

def run_tsga(case_id, seed=None, **overrides):
    """
    Run TSGA on a single case. Returns a full result dict.

    Any hyperparameter in `params.default_params(n)` can be overridden
    via a keyword argument, e.g. `run_tsga("S1", pop_size=200)`.
    """
    matrix = get_instance(case_id)
    n      = len(matrix)
    rng    = np.random.default_rng(seed)

    params = {**default_params(n), **overrides}

    # Bind objectives to plain callables for ga_engine
    fit_fn  = lambda t: fitness(case_id, t, matrix)
    cost_fn = lambda t: tour_cost(t, matrix)

    seeds = get_heuristic_seeds(case_id, n)

    t0  = time.perf_counter()
    out = ga_loop(n, fit_fn, cost_fn, params, rng, seeds=seeds)
    dt  = time.perf_counter() - t0

    ceiling = get_ceiling(case_id)

    return {
        "case_id"         : case_id.upper(),
        "n"               : n,
        "best_fitness"    : int(out["best_fit"]),
        "best_cost"       : int(out["best_cost"]),
        "ceiling"         : int(ceiling),
        "feasible"        : bool(out["best_cost"] <= ceiling),
        "fitness_history" : out["fitness_history"],
        "cost_history"    : out["cost_history"],
        "converged_iter"  : out["converged_iter"],
        "best_solution"   : list(out["best_tour"]),
        "time_s"          : round(dt, 4),
        "params"          : params,
    }


# ---------------- Public API ----------------

def solve(case_id):
    """
    Return just the raw tour cost — drop-in compatible with
    MST_solver.solve and christo_solver.solve.
    """
    return run_tsga(case_id)["best_cost"]


def solve_with_details(case_id, **kwargs):
    """Full result dict. Accepts any TSGA parameter override."""
    return run_tsga(case_id, **kwargs)


def run_on_cases(case_list, seed=None, **overrides):
    """Run TSGA on a list of case IDs. Returns a list of result dicts."""
    return [run_tsga(cid, seed=seed, **overrides) for cid in case_list]