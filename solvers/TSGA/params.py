"""
params.py
=========
Hyperparameter scaling for TSGA.

This file is the ONLY place to tune:
    - population size
    - number of generations
    - crossover rate (pc)
    - mutation rate (pm)
    - Tabu Search budgets and trigger threshold

Sizes are auto-selected by instance size n. To add a new size band,
add another `if n <= X` clause below.
"""


def default_params(n):
    """
    Return a dict of hyperparameters for a TSP instance of size `n`.

    Bands:
        small  : n <= 30
        medium : 30 < n <= 150
        large  : n > 150

    Any key can be overridden per-run by passing it as a keyword to
    `run_tsga(case_id, key=value, ...)`.
    """
    if n <= 30:
        return dict(
            pop_size=60,
            max_gen=150,
            pc=0.85,
            pm=0.15,
            ts_trigger_no_improve=15,
            ts_max_iter=60,
            ts_tenure=10,
            ts_neighborhood_size=30,
        )

    if n <= 150:
        return dict(
            pop_size=100,
            max_gen=250,
            pc=0.85,
            pm=0.10,
            ts_trigger_no_improve=20,
            ts_max_iter=100,
            ts_tenure=15,
            ts_neighborhood_size=40,
        )

    return dict(
        pop_size=120,
        max_gen=400,
        pc=0.85,
        pm=0.07,
        ts_trigger_no_improve=25,
        ts_max_iter=150,
        ts_tenure=20,
        ts_neighborhood_size=50,
    )