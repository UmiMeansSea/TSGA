# ============================================================
#  TSP Instance Generator — SMALL CASE (Taguchi L9, N = 20)
#  Outputs: tsp_small/S1_matrix.csv, S2_matrix.csv, S3_matrix.csv
# ============================================================
import numpy as np
import os, math

def generate_coordinates(n, distribution, rng):
    if distribution == "uniform":
        return rng.uniform(0, 1000, size=(n, 2))

    if distribution == "clustered":
        n_clusters = max(3, n // 7)               # ~3 clusters for n=20
        centers = rng.uniform(0, 1000, size=(n_clusters, 2))
        labels  = rng.integers(0, n_clusters, size=n)
        return centers[labels] + rng.normal(0, 40, size=(n, 2))

    if distribution == "grid":
        side = int(math.ceil(math.sqrt(n)))
        xs, ys = np.meshgrid(np.linspace(0, 1000, side),
                             np.linspace(0, 1000, side))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts + rng.normal(0, 5, size=pts.shape)   # small jitter

    raise ValueError(distribution)

def compute_distance_matrix(coords, metric):
    n, D = len(coords), np.zeros((len(coords), len(coords)), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = coords[i,0] - coords[j,0], coords[i,1] - coords[j,1]
            if metric == "euclidean":
                d = int(round(math.hypot(dx, dy)))
            elif metric == "manhattan":
                d = int(round(abs(dx) + abs(dy)))
            elif metric == "pseudo":                     # PSEUDO-EUCLIDEAN
                d = int(round(math.sqrt((dx*dx + dy*dy) / 10.0)))
            else:
                raise ValueError(metric)
            D[i, j] = D[j, i] = d
    return D

def verify(D, name):
    assert np.all(D == D.T),              f"{name}: not symmetric"
    assert np.all(np.diag(D) == 0),       f"{name}: diagonal not zero"
    assert np.all(D >= 0),                f"{name}: negative distance"
    # triangle inequality spot-check
    n = len(D)
    for _ in range(200):
        i, j, k = np.random.randint(0, n, 3)
        assert D[i,j] <= D[i,k] + D[k,j], f"{name}: triangle violated"
    print(f"  ✔ {name} verified (sym, diag=0, ≥0, triangle ok)")

# --------- Taguchi L9 combinations for the SMALL case ---------
instances = [
    ("S1", 20, "uniform",   "euclidean"),
    ("S2", 20, "clustered", "manhattan"),
    ("S3", 20, "grid",      "pseudo"),
]

os.makedirs("tsp_small", exist_ok=True)
for case_id, n, dist, metric in instances:
    rng    = np.random.default_rng(abs(hash(case_id)) % (2**32))
    coords = generate_coordinates(n, dist, rng)
    D      = compute_distance_matrix(coords, metric)

    np.savetxt(f"tsp_small/{case_id}_matrix.csv", D,      fmt="%d",    delimiter=",")
    np.savetxt(f"tsp_small/{case_id}_coords.csv", coords, fmt="%.4f", delimiter=",")

    print(f"[{case_id}] n={n:>3} | {dist:<9} | {metric:<9} | "
          f"min={D[D>0].min():>4}  max={D.max():>5}  mean={D[D>0].mean():.1f}")
    verify(D, case_id)

print("\nSmall-case instances written to ./tsp_small/")