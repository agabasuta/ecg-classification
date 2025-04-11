import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader, Subset
import numpy as np

from models.models2d import MobileNetV2
from dataloaders.dataset2d import EcgDataset2D
from train import train_one_epoch, validate  # Assuming train.py contains these functions

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 3

# Argument parser to get client_id for each client
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True)
args = parser.parse_args()
CLIENT_ID = args.client_id

# Load dataset and partition it based on the client id
full_dataset = EcgDataset2D("data/train5.json", "data/class-mapper.json")
data_len = len(full_dataset)
indices = np.arange(data_len)
np.random.seed(42)
np.random.shuffle(indices)

partition_size = data_len // NUM_CLIENTS
start = CLIENT_ID * partition_size
end = start + partition_size if CLIENT_ID != NUM_CLIENTS - 1 else data_len
client_subset = Subset(full_dataset, indices[start:end])
train_loader = DataLoader(client_subset, batch_size=32, shuffle=True)

# Initialize model and optimizer
model = MobileNetV2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Flower Client Class
class ECGClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(model.state_dict().keys(), parameters)
        model.load_state_dict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict}, strict=True)

    def fit(self, parameters, config):
        """Train model on local data for one round."""
        # Set parameters for the model
        self.set_parameters(parameters)
        
        # Get the round number from the config (Flower provides it)
        round_num = config.get("round", "N/A")
        print(f"[CLIENT {CLIENT_ID}] Starting training (fit) for round {round_num}")
        
        # Train model for one epoch
        train_one_epoch(model, train_loader, optimizer, DEVICE)
        
        # Return the updated model parameters and other metrics
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local data."""
        # Set parameters for the model
        self.set_parameters(parameters)
        
        # Evaluate on local dataset
        loss, acc = validate(model, train_loader, DEVICE)
        
        # Return loss and accuracy
        return loss, len(train_loader.dataset), {"accuracy": acc}

# Start the Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=ECGClient())
