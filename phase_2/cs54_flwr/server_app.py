"""cs54-flwr: A Flower / PyTorch app."""

import flwr as fl
from flwr.common import Parameters, FitRes, NDArrays, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from cs54_flwr.task import Net, get_weights, weighted_average
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

class SimpleFedAvg(FedAvg):
    """Simple Custom Federated Averaging Strategy."""
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Initialize model to get parameter structure
        self.net = Net()
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights using sample-weighted average."""
        if not results:
            return None, {}
        
        # Extract weights and metrics
        weights_results = []
        total_examples = 0
        weighted_metrics = {"loss": 0.0, "accuracy": 0.0}
        
        for _, fit_res in results:
            try:
                # Convert parameters to NumPy and validate
                params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                if not self._validate_params(params):
                    continue
                
                # Get number of examples
                num_examples = fit_res.num_examples
                total_examples += num_examples
                
                # Accumulate weighted metrics
                for metric_name in ["loss", "accuracy"]:
                    if metric_name in fit_res.metrics:
                        weighted_metrics[metric_name] += fit_res.metrics[metric_name] * num_examples
                
                # Add weights for aggregation
                weights_results.append((params, float(num_examples)))
                
            except Exception as e:
                print(f"Error processing client results: {str(e)}")
                continue
        
        if not weights_results:
            return None, {}
            
        # Aggregate parameters
        aggregated_params = self._aggregate_weights(weights_results)
        if aggregated_params is None:
            return None, {}
        
        # Calculate average metrics
        metrics = {
            name: value / total_examples 
            for name, value in weighted_metrics.items()
        }
        
        return fl.common.ndarrays_to_parameters(aggregated_params), metrics
    
    def _validate_params(self, params: NDArrays) -> bool:
        """Validate parameters structure and types."""
        try:
            # Check if number of layers matches
            if len(params) != len(list(self.net.state_dict().values())):
                return False
            
            # Check each parameter's shape
            state_dict = self.net.state_dict()
            for i, (name, param) in enumerate(state_dict.items()):
                if params[i].shape != param.shape:
                    return False
            return True
        except Exception:
            return False
    
    def _aggregate_weights(self, results: List[Tuple[NDArrays, float]]) -> Optional[NDArrays]:
        """Compute weighted average of weights with proper type handling."""
        if not results:
            return None
        
        # Calculate total weight
        total_weight = sum(weight for _, weight in results)
        if total_weight == 0:
            return None
        
        # Initialize aggregated parameters
        aggregated = [
            np.zeros_like(tensor, dtype=np.float32)
            for tensor in results[0][0]
        ]
        
        # Weighted average for each layer
        for weights, weight in results:
            weight_ratio = weight / total_weight
            for i, layer in enumerate(weights):
                # Ensure float32 type for computation
                layer_float = layer.astype(np.float32)
                aggregated[i] += layer_float * weight_ratio
        
        # Convert back to original types
        state_dict = self.net.state_dict()
        for i, (name, param) in enumerate(state_dict.items()):
            if param.dtype == torch.int64:
                aggregated[i] = np.round(aggregated[i]).astype(np.int64)
            
        return aggregated

def server_fn(context: fl.common.Context):
    """Define server configuration."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    net = Net()
    ndarrays = get_weights(net)
    parameters = fl.common.ndarrays_to_parameters(ndarrays)

    # Define strategy with custom aggregation
    strategy = SimpleFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
