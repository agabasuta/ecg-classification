import flwr as fl
from flwr.server.strategy import FedAvg
import time

# Callback for logging fit results
def log_fit_results(rnd: int, results: list):
    print(f"[SERVER] Round {rnd} finished. Fit results: {results}")

# Callback for logging evaluation results
def log_eval_results(rnd: int, results: list):
    print(f"[SERVER] Round {rnd} finished. Evaluation results: {results}")

# Configuration for each round
def fit_config(rnd: int):
    print(f"[SERVER] Configuring fit for round {rnd}...")
    return {
        "batch_size": 16,  # Define your batch size here
        "local_epochs": 1,  # Number of local epochs to train
    }

def evaluate_config(rnd: int):
    print(f"[SERVER] Configuring evaluation for round {rnd}...")
    return {
        "val_steps": 5,  # Define the number of validation steps
    }

# Define the strategy using FedAvg
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
)

if __name__ == "__main__":
    print("[SERVER] Server is initializing...")
    
    # Manually control the rounds and log results
    for round_num in range(3):  # Example for 3 rounds
        print(f"[SERVER] Starting round {round_num + 1}")
        
        # Start server for this round with the strategy
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(num_rounds=1),  # Run one round at a time
            strategy=strategy
        )
        
        # Log results after each round (modify this to capture actual results)
        # Simulating fit and evaluate results
        log_fit_results(round_num + 1, results=["Sample Fit Results"])
        log_eval_results(round_num + 1, results=["Sample Eval Results"])
