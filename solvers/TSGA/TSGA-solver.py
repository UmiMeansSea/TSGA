"""
tsga_solver.py
==============
TSGA hybrid (Genetic Algorithm + Tabu Search) adapted from WTA to TSP.

Original TSGA (WTA): maximization, M x N integer matrix, row-cap constraint.
This version (TSP):  minimization, permutation of city indices, no constraints.

Public API (matches mst_solver.py / christofides_solver.py):
    solve(case_id)                -> int       (raw tour cost)
    solve_with_details(case_id)   -> dict      (full breakdown)

Fitness is delegated to common/Objective.py, so the penalty rule
(cost + 1000 if cost > 2 * MST_cost) is applied automatically.
"""

import os
import sys
import time
import numpy as np

# ---------------- Path setup ----------------
HERE   = os.path.dirname(os.path.abspath(__file__))         # common/approximator
COMMON = os.path.normpath(os.path.join(HERE, ".."))         # common
if COMMON not in sys.path:
    sys.path.insert(0, COMMON)

from Objective import (
    fitness,
    get_instance,
    get_ceiling,
    tour_cost,
    ALL_CASES,
)

# Optionally seed the initial population with heuristic tours
try:
    from approximator.mst_solver import solve_with_details as _mst_details
    from approximator.christofides_solver import solve_with_details as _chr_details
    _HAS_SEEDS = True
except Exception:
    _HAS_SEEDS = False


# ============================================================
#  GA operators — permutation based
# ============================================================

def _random_tour(n, rng):
    t = list(range(n))
    rng.shuffle(t)
    return t


def _order_crossover(p1, p2, rng):
    """Order Crossover (OX) — a standard permutation crossover."""
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False).tolist())
    child = [-1] * n
    child[a:b + 1] = p1[a:b + 1]
    fill = [x for x in p2 if x not in p1[a:b + 1]]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child


def _swap_mutation(tour, rng):
    """Swap two random positions. Fidelity to TSGA's 'single-gene delta'."""
    t = tour[:]
    i, j = rng.choice(len(t), 2, replace=False).tolist()
    t[i], t[j] = t[j], t[i]
    return t


# ============================================================
#  Tabu Search layer
# ============================================================

def _hash_tour(tour):
    return hash(tuple(tour))


def _two_opt_neighbor(tour, rng):
    """Reverse a random segment [i, j]. Classic 2-opt move."""
    n = len(tour)
    if n < 3:
        return None
    i, j = sorted(rng.choice(n, 2, replace=False).tolist())
    if j - i < 1:
        return None
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


def _tabu_search(start_tour, case_id, matrix, rng,
                 max_ts_iter, tabu_tenure, neighborhood_size):
    """
    Tabu Search on permutations (minimization).
    - Neighborhood: random 2-opt moves.
    - Tabu list: recent tour hashes with tenures.
    - Aspiration: tabu tours allowed if strictly better than global best.
    """
    best_tour = start_tour[:]
    best_fit  = fitness(case_id, best_tour, matrix)

    current_tour = start_tour[:]
    tabu = {}  # hash -> tenure remaining

    for _ in range(max_ts_iter):
        candidates = []

        for _ in range(neighborhood_size):
            cand = _two_opt_neighbor(current_tour, rng)
            if cand is None:
                continue

            h = _hash_tour(cand)
            f = fitness(case_id, cand, matrix)

            # Tabu check (minimization: "not better" means f >= best_fit)
            if h in tabu and tabu[h] > 0 and f >= best_fit:
                continue

            candidates.append((f, cand, h))

        if not candidates:
            break

        # Choose best candidate (MINIMIZATION)
        candidates.sort(key=lambda x: x[0])
        f_cand, cand, h_cand = candidates[0]
        current_tour = cand

        if f_cand < best_fit:
            best_fit  = f_cand
            best_tour = cand[:]

        # Update tabu list
        tabu[h_cand] = tabu_tenure
        for h in list(tabu.keys()):
            tabu[h] -= 1
            if tabu[h] <= 0:
                del tabu[h]

    return best_tour


# ============================================================
#  Main hybrid algorithm
# ============================================================

def _run_tsga(case_id, seed=None, **overrides):
    matrix  = get_instance(case_id)
    n       = len(matrix)
    rng     = np.random.default_rng(seed)

    # ------------- Defaults (auto-scaled by n) -------------
    if n <= 30:
        defaults = dict(pop_size=60,  max_gen=150, pc=0.85, pm=0.15,
                        ts_trigger_no_improve=15, ts_max_iter=60,
                        ts_tenure=10, ts_neighborhood_size=30)
    elif n <= 150:
        defaults = dict(pop_size=100, max_gen=250, pc=0.85, pm=0.10,
                        ts_trigger_no_improve=20, ts_max_iter=100,
                        ts_tenure=15, ts_neighborhood_size=40)
    else:
        defaults = dict(pop_size=120, max_gen=400, pc=0.85, pm=0.07,
                        ts_trigger_no_improve=25, ts_max_iter=150,
                        ts_tenure=20, ts_neighborhood_size=50)

    params = {**defaults, **overrides}

    # ------------- Initial population -------------
    pop = [_random_tour(n, rng) for _ in range(params["pop_size"])]

    # Seed with heuristic tours if available (one MST, one Christofides)
    if _HAS_SEEDS and n >= 30:
        try:
            pop[0] = list(_mst_details(case_id)["tour"][:-1])  # drop closing dup
            pop[1] = list(_chr_details(case_id)["tour"][:-1])
        except Exception:
            pass  # fall back to random if seeding fails

    vals = [fitness(case_id, t, matrix) for t in pop]

    best_val  = min(vals)
    best_idx  = vals.index(best_val)
    best_tour = pop[best_idx][:]
    best_cost = tour_cost(best_tour, matrix)

    fitness_hist = [best_val]
    cost_hist    = [best_cost]
    no_improve   = 0

    t0 = time.perf_counter()

    for gen in range(params["max_gen"]):

        # ---------- Selection: binary tournament ----------
        selected = []
        for _ in range(params["pop_size"]):
            a, b = rng.integers(0, params["pop_size"], size=2).tolist()
            winner = a if vals[a] <= vals[b] else b
            selected.append(pop[winner][:])
        pop = selected

        # ---------- Crossover (OX, fixed pc) ----------
        idx = rng.permutation(params["pop_size"]).tolist()
        pop = [pop[i] for i in idx]
        for i in range(0, params["pop_size"] - 1, 2):
            if rng.random() < params["pc"]:
                ca = _order_crossover(pop[i],     pop[i + 1], rng)
                cb = _order_crossover(pop[i + 1], pop[i],     rng)
                pop[i], pop[i + 1] = ca, cb

        # ---------- Mutation (swap, fixed pm) ----------
        for i in range(params["pop_size"]):
            if rng.random() < params["pm"]:
                pop[i] = _swap_mutation(pop[i], rng)

        # ---------- Evaluation ----------
        vals = [fitness(case_id, t, matrix) for t in pop]
        cur       = min(vals)
        cur_idx   = vals.index(cur)

        if cur < best_val:
            best_val  = cur
            best_tour = pop[cur_idx][:]
            best_cost = tour_cost(best_tour, matrix)
            no_improve = 0
        else:
            no_improve += 1

        fitness_hist.append(best_val)
        cost_hist.append(best_cost)

        # ---------- Tabu Search trigger ----------
        if no_improve >= params["ts_trigger_no_improve"]:
            ts_best = _tabu_search(
                best_tour, case_id, matrix, rng,
                params["ts_max_iter"],
                params["ts_tenure"],
                params["ts_neighborhood_size"],
            )
            ts_fit = fitness(case_id, ts_best, matrix)

            # Replace worst individual with TS result
            worst_idx = vals.index(max(vals))
            pop[worst_idx]  = ts_best[:]
            vals[worst_idx] = ts_fit

            if ts_fit < best_val:
                best_val  = ts_fit
                best_tour = ts_best[:]
                best_cost = tour_cost(best_tour, matrix)

            no_improve = 0

    elapsed = time.perf_counter() - t0
    ceiling = get_ceiling(case_id)

    return {
        "case_id"         : case_id.upper(),
        "n"               : n,
        "best_fitness"    : int(best_val),
        "best_cost"       : int(best_cost),
        "ceiling"         : int(ceiling),
        "feasible"        : bool(best_cost <= ceiling),
        "fitness_history" : fitness_hist,
        "cost_history"    : cost_hist,
        "converged_iter"  : int(np.argmin(fitness_hist)),
        "best_solution"   : list(best_tour),
        "time_s"          : round(elapsed, 4),
    }


# ============================================================
#  Public API — same shape as MST and Christofides solvers
# ============================================================

def solve(case_id):
    """
    Run TSGA on the generated instance `case_id`.

    Parameters
    ----------
    case_id : str
        One of 'S1','S2','S3','M1','M2','M3','L1','L2','L3'.

    Returns
    -------
    int
        Raw tour cost (length) of the best tour found.
        Matches the return type of mst_solver.solve / christofides_solver.solve.
    """
    return _run_tsga(case_id)["best_cost"]


def solve_with_details(case_id, **kwargs):
    """
    Same as solve() but returns the full dict — history, feasibility,
    convergence, best_solution, runtime. Any of the TSGA parameters can
    be overridden via keyword arguments.
    """
    return _run_tsga(case_id, **kwargs)


# ---------------- Smoke test ----------------
if __name__ == "__main__":
    for cid in ALL_CASES:
        r = _run_tsga(cid)
        tag = "OK" if r["feasible"] else "PENALIZED"
        print(f"{cid:>3} | n={r['n']:>3} "
              f"| cost={r['best_cost']:>9} "
              f"| ceiling={r['ceiling']:>9} "
              f"| {tag:<10} "
              f"| conv@{r['converged_iter']:>4} "
              f"| t={r['time_s']:.2f}s")