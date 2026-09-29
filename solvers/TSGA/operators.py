"""
operators.py
============
All GA operators that manipulate a single tour or a whole population.

Contains:
    - random_tour          : initial tour generator
    - order_crossover      : OX crossover (permutation-safe)
    - swap_mutation        : swap two positions
    - two_opt_neighbor     : 2-opt move (used by Tabu Search)
    - tournament_selection : binary tournament (minimization)
    - crossover_population : apply OX across the pop
    - mutate_population    : apply swap mutation across the pop

This file has NO knowledge of cases, files, or fitness. It only
manipulates lists of city indices.

To change an operator: edit the corresponding function here.
"""
import numpy as np


# ── Single-tour operators ────────────────────────────────────

def random_tour(n, rng):
    """Return a random permutation of [0..n-1]."""
    t = list(range(n))
    rng.shuffle(t)
    return t


def order_crossover(p1, p2, rng):
    """Order Crossover (OX) — produces a valid permutation child."""
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


def swap_mutation(tour, rng):
    """Swap two random positions."""
    t = tour[:]
    i, j = rng.choice(len(t), 2, replace=False).tolist()
    t[i], t[j] = t[j], t[i]
    return t


def two_opt_neighbor(tour, rng):
    """Reverse a random segment [i, j]. Returns None if n < 3."""
    n = len(tour)
    if n < 3:
        return None
    i, j = sorted(rng.choice(n, 2, replace=False).tolist())
    if j - i < 1:
        return None
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


# ── Population-level operators ──────────────────────────────

def tournament_selection(pop, vals, pop_size, rng):
    """
    Binary tournament. Minimization: smaller `vals` wins.
    Returns a new list of pop_size tours (copies).
    """
    selected = []
    for _ in range(pop_size):
        a, b = rng.integers(0, len(pop), 2).tolist()
        winner = a if vals[a] <= vals[b] else b
        selected.append(pop[winner][:])
    return selected


def crossover_population(pop, pc, rng):
    """In-place OX crossover on a random pairing of the population."""
    idx = rng.permutation(len(pop)).tolist()
    pop = [pop[i] for i in idx]
    for i in range(0, len(pop) - 1, 2):
        if rng.random() < pc:
            ca = order_crossover(pop[i],     pop[i + 1], rng)
            cb = order_crossover(pop[i + 1], pop[i],     rng)
            pop[i], pop[i + 1] = ca, cb
    return pop


def mutate_population(pop, pm, rng):
    """Apply swap mutation to each individual with probability pm."""
    for i in range(len(pop)):
        if rng.random() < pm:
            pop[i] = swap_mutation(pop[i], rng)
    return pop