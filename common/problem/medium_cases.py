# ============================================================
#  TSP Instance Generator — MEDIUM CASE (Taguchi L9, N = 100)
#  Outputs: tsp_medium/M1_matrix.csv, M2_matrix.csv, M3_matrix.csv
# ============================================================
import numpy as np
import os, math

def generate_coordinates(n, distribution, rng):
    if distribution == "uniform":
        return rng.uniform(0, 2000, size=(n, 2))

    if distribution == "clustered":
        n_clusters = max(5, n // 15)              # ~7 clusters for n=100
        centers = rng.uniform(0, 2000, size=(n_clusters, 2))
        labels  = rng.integers(0, n_clusters, size=n)
        return centers[labels] + rng.normal(0, 60, size=(n, 2))

    if distribution == "grid":
        side = int(math.ceil(math.sqrt(n)))       # 10 x 10 for n=100
        xs, ys = np.meshgrid(np.linspace(0, 2000, side),
                             np.linspace(0, 2000, side))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts + rng.normal(0, 8, size=pts.shape)

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
            elif metric == "pseudo":
                d = int(round(math.sqrt((dx*dx + dy*dy) / 10.0)))
            else:
                raise ValueError(metric)
            D[i, j] = D[j, i] = d
    return D

def verify(D, name):
    assert np.all(D == D.T),        f"{name}: not symmetric"
    assert np.all(np.diag(D) == 0), f"{name}: diagonal not zero"
    assert np.all(D >= 0),          f"{name}: negative distance"
    n = len(D)
    for _ in range(500):
        i, j, k = np.random.randint(0, n, 3)
        assert D[i,j] <= D[i,k] + D[k,j], f"{name}: triangle violated"
    print(f"  ✔ {name} verified")

# --------- Taguchi L9 combinations for the MEDIUM case ---------
instances = [
    ("M1", 100, "uniform",   "manhattan"),
    ("M2", 100, "clustered", "pseudo"),
    ("M3", 100, "grid",      "euclidean"),
]

os.makedirs("tsp_medium", exist_ok=True)
for case_id, n, dist, metric in instances:
    rng    = np.random.default_rng(abs(hash(case_id)) % (2**32))
    coords = generate_coordinates(n, dist, rng)
    D      = compute_distance_matrix(coords, metric)

    np.savetxt(f"tsp_medium/{case_id}_matrix.csv", D,      fmt="%d",    delimiter=",")
    np.savetxt(f"tsp_medium/{case_id}_coords.csv", coords, fmt="%.4f", delimiter=",")

    print(f"[{case_id}] n={n:>3} | {dist:<9} | {metric:<9} | "
          f"min={D[D>0].min():>4}  max={D.max():>6}  mean={D[D>0].mean():.1f}")
    verify(D, case_id)

print("\nMedium-case instances written to ./tsp_medium/")