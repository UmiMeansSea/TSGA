import torch
import torch.optim as optim
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the user's provided objective function contract
from common.objectives import get_instance, evaluate, ALL_CASES
from dcnn_model import TSPDCNN

def train_and_evaluate(epochs=100, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSPDCNN(embedding_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ALL_CASES contains ['S1', 'S2', 'S3', 'M1', 'M2', 'M3', 'L1', 'L2', 'L3']
    for case_id in ALL_CASES:
        print(f"\n--- Initializing Solver for Case: {case_id} ---")
        
        # Load distance matrix from objective.py cache
        raw_matrix = get_instance(case_id)
        
        # Convert to batch format for PyTorch: (1, n, n)
        matrix_tensor = torch.tensor(raw_matrix, dtype=torch.float32).unsqueeze(0).to(device)

        # ---------------------------------------------------------
        # Training Phase (RL Improvement)
        # ---------------------------------------------------------
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass: Get tour and log probabilities
            tour_tensor, log_probs = model(matrix_tensor, return_pi=True)
            
            # Convert tour back to standard Python list for objective.py contract
            tour_list = tour_tensor[0].cpu().numpy().tolist()
            
            # Get cost using objective.py validation (evaluates closed cycle)
            eval_data = evaluate(case_id, tour_list, raw_matrix)
            cost = eval_data["cost"]
            
            # Baseline approximation (Simple Moving Average) for variance reduction
            baseline = eval_data["ceiling"] / 2.0 
            
            # REINFORCE loss: -log_prob * (reward - baseline)
            # We want to minimize cost, so reward is negative cost
            reward = -float(cost)
            advantage = reward - (-baseline) 
            
            loss = -log_probs.mean() * advantage
            
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch:3d} | Raw Cost: {cost} | Loss: {loss.item():.4f}")

        # ---------------------------------------------------------
        # Evaluation Phase (Constructive Policy)
        # ---------------------------------------------------------
        model.eval()
        with torch.no_grad():
            final_tour_tensor = model(matrix_tensor, return_pi=False)
            final_tour_list = final_tour_tensor[0].cpu().numpy().tolist()
            
            # Send to objective.py for penalized fitness evaluation
            result = evaluate(case_id, final_tour_list, raw_matrix)
            
            print(f"RESULTS FOR {case_id}:")
            print(f"  Nodes (N) : {result['n']}")
            print(f"  Raw Cost  : {result['cost']}")
            print(f"  Ceiling   : {result['ceiling']}")
            print(f"  Penalty   : {result['penalty']}")
            print(f"  FITNESS   : {result['fitness']}")
            print(f"  Status    : {'PENALIZED' if result['penalized'] else 'ACCEPTED'}")

if __name__ == "__main__":
    # Ensure reproducibility for testing the dataset
    torch.manual_seed(42)
    train_and_evaluate(epochs=150, learning_rate=5e-4)