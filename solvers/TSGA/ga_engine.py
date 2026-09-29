"""
ga_engine.py
============
The main GA + Tabu Search loop.

Contains:
    - ga_loop

`ga_loop` orchestrates a single TSGA run:
    1. Build initial population (via population.py)
    2. Evaluate
    3. For each generation:
         a. select  (operators.tournament_selection)
         b. crossover (operators.crossover_population)
         c. mutate   (operators.mutate_population)
         d. evaluate (population.evaluate_population)
         e. update best / stagnation counter
         f. if stagnation >= threshold, run Tabu Search and inject
    4. Return best tour and histories

To change when TS triggers, what it returns to the pop, or how
stagnation is measured: edit only this file.
"""
import numpy as np

from .operators import (
    tournament_selection,
    crossover_population,
    mutate_population,
)
from .population import (
    build_initial_population,
    evaluate_population,
)
from .tabu_search import tabu_search


def ga_loop(n, fit_fn, cost_fn, params, rng, seeds=None):
    """
    Run one full TSGA experiment.

    Parameters
    ----------
    n       : int                         number of cities
    fit_fn  : callable(tour) -> float     penalized fitness (minimize)
    cost_fn : callable(tour) -> int       raw tour cost
    params  : dict                        from params.default_params(n)
    rng     : np.random.Generator
    seeds   : list[tour] or None          heuristic seeds

    Returns
    -------
    dict:
        best_tour, best_fit, best_cost,
        fitness_history, cost_history, converged_iter
    """
    pop  = build_initial_population(n, params["pop_size"], rng, seeds=seeds)
    vals = evaluate_population(pop, fit_fn)

    best_idx  = int(np.argmin(vals))
    best_tour = pop[best_idx][:]
    best_fit  = vals[best_idx]
    best_cost = cost_fn(best_tour)

    fitness_hist = [best_fit]
    cost_hist    = [best_cost]
    no_improve   = 0

    for _ in range(params["max_gen"]):

        # ── GA step ─────────────────────────────────────────
        pop = tournament_selection(pop, vals, params["pop_size"], rng)
        pop = crossover_population(pop, params["pc"], rng)
        pop = mutate_population(pop, params["pm"], rng)

        vals = evaluate_population(pop, fit_fn)
        cur  = min(vals)

        if cur < best_fit:
            cur_idx    = vals.index(cur)
            best_tour  = pop[cur_idx][:]
            best_fit   = cur
            best_cost  = cost_fn(best_tour)
            no_improve = 0
        else:
            no_improve += 1

        fitness_hist.append(best_fit)
        cost_hist.append(best_cost)

        # ── Tabu Search trigger ─────────────────────────────
        if no_improve >= params["ts_trigger_no_improve"]:
            ts_best = tabu_search(
                best_tour, fit_fn, rng,
                params["ts_max_iter"],
                params["ts_tenure"],
                params["ts_neighborhood_size"],
            )
            ts_fit = fit_fn(ts_best)

            # Inject into population by replacing the worst individual
            worst_idx = vals.index(max(vals))
            pop[worst_idx]  = ts_best[:]
            vals[worst_idx] = ts_fit

            if ts_fit < best_fit:
                best_fit  = ts_fit
                best_tour = ts_best[:]
                best_cost = cost_fn(best_tour)

            no_improve = 0

    return {
        "best_tour"       : best_tour,
        "best_fit"        : best_fit,
        "best_cost"       : best_cost,
        "fitness_history" : fitness_hist,
        "cost_history"    : cost_hist,
        "converged_iter"  : int(np.argmin(fitness_hist)),
    }