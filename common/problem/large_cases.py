# ============================================================
#  TSP Instance Generator — LARGE CASE (Taguchi L9, N = 500)
#  Outputs: tsp_large/L1_matrix.csv, L2_matrix.csv, L3_matrix.csv
# ============================================================
import numpy as np
import os, math

def generate_coordinates(n, distribution, rng):
    if distribution == "uniform":
        return rng.uniform(0, 5000, size=(n, 2))

    if distribution == "clustered":
        n_clusters = max(10, n // 25)             # ~20 clusters for n=500
        centers = rng.uniform(0, 5000, size=(n_clusters, 2))
        labels  = rng.integers(0, n_clusters, size=n)
        return centers[labels] + rng.normal(0, 100, size=(n, 2))

    if distribution == "grid":
        side = int(math.ceil(math.sqrt(n)))       # ~23 x 23 for n=500
        xs, ys = np.meshgrid(np.linspace(0, 5000, side),
                             np.linspace(0, 5000, side))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts + rng.normal(0, 15, size=pts.shape)

    raise ValueError(distribution)

def compute_distance_matrix(coords, metric):
    """Vectorised — fast enough for 500×500."""
    diff = coords[:, None, :] - coords[None, :, :]
    if metric == "euclidean":
        D = np.sqrt((diff ** 2).sum(-1))
    elif metric == "manhattan":
        D = np.abs(diff).sum(-1)
    elif metric == "pseudo":
        D = np.sqrt((diff ** 2).sum(-1) / 10.0)
    else:
        raise ValueError(metric)
    D = np.round(D).astype(int)
    np.fill_diagonal(D, 0)
    D = np.triu(D) + np.triu(D, 1).T      # enforce exact symmetry
    return D

def verify(D, name):
    assert np.all(D == D.T),        f"{name}: not symmetric"
    assert np.all(np.diag(D) == 0), f"{name}: diagonal not zero"
    assert np.all(D >= 0),          f"{name}: negative distance"
    n = len(D)
    for _ in range(1000):
        i, j, k = np.random.randint(0, n, 3)
        assert D[i,j] <= D[i,k] + D[k,j], f"{name}: triangle violated"
    print(f"  ✔ {name} verified")

# --------- Taguchi L9 combinations for the LARGE case ---------
instances = [
    ("L1", 500, "uniform",   "pseudo"),
    ("L2", 500, "clustered", "euclidean"),
    ("L3", 500, "grid",      "manhattan"),
]

os.makedirs("tsp_large", exist_ok=True)
for case_id, n, dist, metric in instances:
    rng    = np.random.default_rng(abs(hash(case_id)) % (2**32))
    coords = generate_coordinates(n, dist, rng)
    D      = compute_distance_matrix(coords, metric)

    np.savetxt(f"tsp_large/{case_id}_matrix.csv", D,      fmt="%d",    delimiter=",")
    np.savetxt(f"tsp_large/{case_id}_coords.csv", coords, fmt="%.4f", delimiter=",")

    print(f"[{case_id}] n={n:>3} | {dist:<9} | {metric:<9} | "
          f"min={D[D>0].min():>4}  max={D.max():>6}  mean={D[D>0].mean():.1f}")
    verify(D, case_id)

print("\nLarge-case instances written to ./tsp_large/")