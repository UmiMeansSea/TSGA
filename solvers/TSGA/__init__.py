"""
solvers.TSGA
============
TSGA (Genetic Algorithm + Tabu Search) for TSP.

Public entry point:
    from solvers.TSGA.main import solve, solve_with_details, run_on_cases
"""
from .main import solve, solve_with_details, run_tsga, run_on_cases

__all__ = ["solve", "solve_with_details", "run_tsga", "run_on_cases"]