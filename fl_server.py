import flwr as fl
from flwr.server.strategy import FedAvg

strategy = FedAvg(
    fraction_fit=1.0,
    min_fit_clients=3,
    min_eval_clients=3,
    min_available_clients=3,
)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config={"num_rounds": 5},
    strategy=strategy,
)
