import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_main_results():
    if not os.path.exists("results_main.csv"):
        print("results_main.csv not found")
        return
    
    df = pd.read_csv("results_main.csv")
    
    # Cost Comparison
    plt.figure(figsize=(10, 6))
    for solver in df['Solver'].unique():
        solver_df = df[df['Solver'] == solver]
        plt.plot(solver_df['Case'], solver_df['Generated_Cost'], marker='o', label=solver)
    
    plt.title("Generated Cost by Solver and Case")
    plt.xlabel("Case")
    plt.ylabel("Generated Cost")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_main_cost.png")
    plt.close()

    # Time Comparison
    plt.figure(figsize=(10, 6))
    for solver in df['Solver'].unique():
        solver_df = df[df['Solver'] == solver]
        plt.plot(solver_df['Case'], solver_df['Run_Time_s'], marker='x', label=solver)
        
    plt.title("Run Time (s) by Solver and Case")
    plt.xlabel("Case")
    plt.ylabel("Run Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph_main_time.png")
    plt.close()

def plot_fitness_history():
    if not os.path.exists("results_fitness_history.csv"):
        print("results_fitness_history.csv not found")
        return
        
    df = pd.read_csv("results_fitness_history.csv")
    
    for _, row in df.iterrows():
        case = row['Case']
        solver = row['Solver']
        history_str = row['Fitness_History']
        if pd.isna(history_str): continue
        history = [float(x) for x in history_str.split(',')]
        
        plt.figure(figsize=(8, 5))
        plt.plot(history, label=f"{solver} on {case}")
        plt.title(f"Fitness History - {solver} - {case}")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"graph_history_{solver}_{case}.png")
        plt.close()

def plot_christofides():
    if not os.path.exists("results_christofides.csv"):
        print("results_christofides.csv not found")
        return
        
    df = pd.read_csv("results_christofides.csv")
    
    plt.figure(figsize=(10, 6))
    plt.bar(df['Case'], df['Generated_Cost'], color='orange', alpha=0.7)
    plt.title("Christofides Baseline Cost by Case")
    plt.xlabel("Case")
    plt.ylabel("Generated Cost")
    plt.grid(axis='y')
    plt.savefig("graph_christofides_cost.png")
    plt.close()

if __name__ == "__main__":
    plot_main_results()
    plot_fitness_history()
    plot_christofides()
    print("Graphs have been generated and saved as PNG files.")
