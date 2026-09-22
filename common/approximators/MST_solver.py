# ============================================================
#  MST Double-Tree Solver — callable by case ID
#  Usage:  from mst_solver import solve
#          cost = solve("S1")     # -> int
# ============================================================
import os
import math
import numpy as np

# ---------------- Paths ----------------
HERE        = os.path.dirname(os.path.abspath(__file__))
PROBLEM_DIR = os.path.normpath(os.path.join(HERE, "..", "problem"))

CASE_FOLDER = {
    "S": "tsp_small",
    "M": "tsp_medium",
    "L": "tsp_large",
}

# ---------------- I/O ----------------
def _load_matrix(case_id):
    """Load the distance matrix for a case like 'S1', 'M2', 'L3'."""
    case_id = case_id.strip().upper()
    if not case_id or case_id[0] not in CASE_FOLDER:
        raise ValueError(f"Unknown case id: {case_id!r}. "
                         f"Expected S1..S3, M1..M3, L1..L3.")
    folder = CASE_FOLDER[case_id[0]]
    path   = os.path.join(PROBLEM_DIR, folder, f"{case_id}_matrix.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Matrix not found for {case_id}: {path}\n"
            f"Did you run the generator for {folder}?"
        )
    return np.loadtxt(path, delimiter=",", dtype=int)

# ---------------- Core building blocks ----------------
def _prim_mst(matrix):
    """O(n^2) Prim. Returns list of (u, v) MST edges."""
    n = len(matrix)
    in_tree = [False] * n
    key     = [math.inf] * n
    parent  = [-1] * n
    key[0]  = 0

    for _ in range(n):
        u, best = -1, math.inf
        for i in range(n):
            if not in_tree[i] and key[i] < best:
                best, u = key[i], i
        in_tree[u] = True
        row = matrix[u]
        for v in range(n):
            if not in_tree[v] and row[v] < key[v]:
                key[v] = row[v]
                parent[v] = u

    return [(parent[v], v) for v in range(1, n)]

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _eulerian_circuit(n, adj):
    """Hierholzer's algorithm on a graph with all-even degrees."""
    adj = [list(neigh) for neigh in adj]
    stack, circuit = [0], []
    while stack:
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            adj[v].remove(u)
            stack.append(v)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return circuit

def _shortcut(circuit):
    """Skip revisits to get a Hamiltonian cycle."""
    seen, tour = set(), []
    for v in circuit:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    tour.append(tour[0])
    return tour

def _tour_length(tour, matrix):
    return sum(int(matrix[tour[i]][tour[i + 1]]) for i in range(len(tour) - 1))

# ---------------- Main callable ----------------
def solve(case_id):
    """
    Run the MST double-tree heuristic on the generated instance `case_id`.

    Parameters
    ----------
    case_id : str
        One of 'S1','S2','S3','M1','M2','M3','L1','L2','L3'.

    Returns
    -------
    int
        Cost (total length) of the double-tree tour.
    """
    matrix = _load_matrix(case_id)

    n         = len(matrix)
    mst_edges = _prim_mst(matrix)

    # Double every MST edge so every vertex has even degree
    doubled   = mst_edges + mst_edges
    adj       = _build_adjacency(n, doubled)

    circuit   = _eulerian_circuit(n, adj)
    tour      = _shortcut(circuit)

    return _tour_length(tour, matrix)


def solve_with_details(case_id):
    """
    Same as solve() but also returns intermediate values.
    Useful if you later want to log MST weight or the tour itself.
    """
    matrix    = _load_matrix(case_id)
    n         = len(matrix)
    mst_edges = _prim_mst(matrix)
    doubled   = mst_edges + mst_edges
    adj       = _build_adjacency(n, doubled)
    circuit   = _eulerian_circuit(n, adj)
    tour      = _shortcut(circuit)

    return {
        "case_id"    : case_id.upper(),
        "n"          : n,
        "mst_weight" : sum(int(matrix[u][v]) for u, v in mst_edges),
        "tour_length": _tour_length(tour, matrix),
        "tour"       : tour,
    }