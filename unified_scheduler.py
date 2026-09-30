import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import time
import csv

# Add current dir to path to find common and solvers modules
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common.objectives import get_instance, evaluate, ALL_CASES
from solvers.TSGA.main import solve_with_details


# =============================================================================
# CMIP Model Definition (DCNN + RL)
# =============================================================================
class TSPDCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TSPDCNN, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=embedding_dim, kernel_size=3, padding=1)
        
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, distance_matrix, return_pi=False):
        batch_size, n, _ = distance_matrix.shape
        device = distance_matrix.device

        x = distance_matrix.view(batch_size * n, 1, n).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        node_embeddings = x.mean(dim=2).view(batch_size, n, self.embedding_dim) 

        tours = []
        log_probs = torch.zeros(batch_size, device=device)
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        
        current_city = torch.zeros(batch_size, dtype=torch.long, device=device)
        tours.append(current_city)
        mask[torch.arange(batch_size), current_city] = True

        for step in range(1, n):
            query = node_embeddings[torch.arange(batch_size), current_city]
            query = self.W_q(query).unsqueeze(1)
            
            keys = self.W_k(node_embeddings)
            
            scores = torch.sum(self.v * torch.tanh(query + keys), dim=2)
            scores = scores.masked_fill(mask, float('-inf'))
            
            probs = F.softmax(scores, dim=1)
            
            if self.training:
                m = torch.distributions.Categorical(probs)
                next_city = m.sample()
                log_probs += m.log_prob(next_city)
            else:
                next_city = probs.argmax(dim=1)
                
            tours.append(next_city)
            mask = mask.clone()
            mask[torch.arange(batch_size), next_city] = True
            current_city = next_city

        tour_tensor = torch.stack(tours, dim=1)
        
        if return_pi:
            return tour_tensor, log_probs
        return tour_tensor

def run_cmip_with_details(case_id, matrix, epochs=150, learning_rate=5e-4):
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSPDCNN(embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    matrix_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
    
    fitness_history = []
    converged_epoch = 0
    best_cost = float('inf')
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tour_tensor, log_probs = model(matrix_tensor, return_pi=True)
        tour_list = tour_tensor[0].cpu().numpy().tolist()
        
        eval_data = evaluate(case_id, tour_list, matrix)
        cost = eval_data["cost"]
        fitness_history.append(cost)
        if cost < best_cost:
            best_cost = cost
            converged_epoch = epoch
            
        baseline = eval_data["ceiling"] / 2.0 
        
        reward = -float(cost)
        advantage = reward - (-baseline) 
        
        loss = -log_probs.mean() * advantage
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_tour_tensor = model(matrix_tensor, return_pi=False)
        final_tour_list = final_tour_tensor[0].cpu().numpy().tolist()
        
    dt = time.perf_counter() - t0
    final_eval = evaluate(case_id, final_tour_list, matrix)
    
    return {
        "best_cost": final_eval["cost"],
        "time_s": dt,
        "fitness_history": fitness_history,
        "converged_iter": converged_epoch,
        "max_iter": epochs,
        "best_tour": final_tour_list
    }


# =============================================================================
# Christofides Baseline
# =============================================================================
def run_christofides(matrix):
    G = nx.from_numpy_array(matrix)
    cycle = nx.approximation.traveling_salesman_problem(
        G, weight='weight', cycle=True, method=nx.approximation.christofides
    )
    
    # Short-circuit the cycle to ensure exactly one visit per node (valid permutation)
    # This prevents errors if networkx returns multiple repeated nodes for non-metric graphs
    tour = []
    seen = set()
    for node in cycle:
        if node not in seen:
            seen.add(node)
            tour.append(node)
            
    return tour


# =============================================================================
# Job Scheduler
# =============================================================================
def main(cases_to_run=None):
    if cases_to_run is None:
        cases_to_run = ALL_CASES

    print("==========================================================")
    print("TSP Unified Job Scheduler")
    print(f"Scheduled Cases: {cases_to_run}")
    print("==========================================================")

    # To ensure deterministic behavior for CMIP testing
    torch.manual_seed(42)

    main_results = []
    history_results = []
    christofides_results = []

    for case_id in cases_to_run:
        print(f"\n>>> Executing Job: {case_id} <<<")
        try:
            matrix = get_instance(case_id)
        except Exception as e:
            print(f"  Error loading case {case_id}: {e}")
            continue

        # 1. TSGA
        try:
            tsga_res = solve_with_details(case_id)
            tsga_max_iter = tsga_res["params"]["max_gen"]
            # Convergence normalized
            tsga_conv_norm = tsga_res["converged_iter"] / tsga_max_iter if tsga_max_iter > 0 else 0
            
            main_results.append({
                "Case": case_id,
                "Solver": "TSGA",
                "Run_Time_s": tsga_res["time_s"],
                "Convergence_Normalized": round(tsga_conv_norm, 4),
                "Max_Iterations": tsga_max_iter,
                "Generated_Cost": tsga_res["best_cost"]
            })
            
            history_results.append({
                "Case": case_id,
                "Solver": "TSGA",
                "Fitness_History": ",".join(map(str, tsga_res["fitness_history"]))
            })
            
            print(f"  [TSGA] Cost: {tsga_res['best_cost']} | Time: {tsga_res['time_s']:.2f}s")
        except Exception as e:
            print(f"  [TSGA] Error: {e}")

        # 2. CMIP
        try:
            cmip_res = run_cmip_with_details(case_id, matrix)
            cmip_max_iter = cmip_res["max_iter"]
            cmip_conv_norm = cmip_res["converged_iter"] / cmip_max_iter if cmip_max_iter > 0 else 0
            
            main_results.append({
                "Case": case_id,
                "Solver": "CMIP",
                "Run_Time_s": round(cmip_res["time_s"], 4),
                "Convergence_Normalized": round(cmip_conv_norm, 4),
                "Max_Iterations": cmip_max_iter,
                "Generated_Cost": cmip_res["best_cost"]
            })
            
            history_results.append({
                "Case": case_id,
                "Solver": "CMIP",
                "Fitness_History": ",".join(map(str, cmip_res["fitness_history"]))
            })
            print(f"  [CMIP] Cost: {cmip_res['best_cost']} | Time: {cmip_res['time_s']:.2f}s")
        except Exception as e:
            print(f"  [CMIP] Error: {e}")

        # 3. Christofides
        try:
            t0 = time.perf_counter()
            christo_tour = run_christofides(matrix)
            dt = time.perf_counter() - t0
            christo_res = evaluate(case_id, christo_tour, matrix)
            
            christofides_results.append({
                "Case": case_id,
                "Run_Time_s": round(dt, 4),
                "Generated_Cost": christo_res["cost"],
                "Solution_Tour": ",".join(map(str, christo_tour))
            })
            print(f"  [Christofides] Cost: {christo_res['cost']} | Time: {dt:.4f}s")
        except Exception as e:
            print(f"  [Christofides] Error: {e}")

    # Write Results to CSVs
    print("\nWriting results to CSV files...")
    
    with open("results_main.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Case", "Solver", "Run_Time_s", "Convergence_Normalized", "Max_Iterations", "Generated_Cost"])
        writer.writeheader()
        writer.writerows(main_results)

    with open("results_fitness_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Case", "Solver", "Fitness_History"])
        writer.writeheader()
        writer.writerows(history_results)
        
    with open("results_christofides.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Case", "Run_Time_s", "Generated_Cost", "Solution_Tour"])
        writer.writeheader()
        writer.writerows(christofides_results)

    print("All tasks finished. Results saved to: results_main.csv, results_fitness_history.csv, results_christofides.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        valid_args = [c.upper() for c in args if c.upper() in ALL_CASES]
        if not valid_args:
            print("No valid cases provided. Running default ALL_CASES.")
            main()
        else:
            main(valid_args)
    else:
        main()
