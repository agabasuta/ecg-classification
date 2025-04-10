from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np

def get_federated_partition(dataset, client_id, num_clients=3):
    data_len = len(dataset)
    indices = np.arange(data_len)
    np.random.seed(42)
    np.random.shuffle(indices)

    partition_size = data_len // num_clients
    start = client_id * partition_size
    end = start + partition_size if client_id != num_clients - 1 else data_len
    return Subset(dataset, indices[start:end])
