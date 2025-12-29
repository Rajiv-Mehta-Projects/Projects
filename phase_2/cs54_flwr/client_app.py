"""cs54-flwr: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from cs54_flwr.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs):
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize net after data is loaded to get correct number of classes
        self.net = None

    def fit(self, parameters, config):
        # Initialize net if not already done
        if self.net is None:
            self.net = Net()
            self.net.to(self.device)
        
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        # Initialize net if not already done
        if self.net is None:
            self.net = Net()
            self.net.to(self.device)
            
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load data first to determine number of classes
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
