import flwr as fl
import torch
import argparse
from torch.utils.data import DataLoader, Subset
import numpy as np

from models.models2d import MobileNetV2
from dataloaders.dataset2d import CustomDataset  # Uses 2D ECG images
from train import train_one_epoch, validate  # Use the train functions from your repo

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 3

# Argparse to get client_id
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True)
args = parser.parse_args()
CLIENT_ID = args.client_id

# Load dataset and split
full_dataset = CustomDataset(mode="train")
data_len = len(full_dataset)
indices = np.arange(data_len)
np.random.seed(42)
np.random.shuffle(indices)

partition_size = data_len // NUM_CLIENTS
start = CLIENT_ID * partition_size
end = start + partition_size if CLIENT_ID != NUM_CLIENTS - 1 else data_len
client_subset = Subset(full_dataset, indices[start:end])
train_loader = DataLoader(client_subset, batch_size=32, shuffle=True)

# Init model
model = get_model("mobilenetv2").to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Flower Client
class ECGClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        model.load_state_dict({k: torch.tensor(v).to(DEVICE) for k, v in params_dict}, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_one_epoch(model, train_loader, optimizer, DEVICE)
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = validate(model, train_loader, DEVICE)
        return loss, len(train_loader.dataset), {"accuracy": acc}

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=ECGClient())
