# -*- coding: utf-8 -*-
"""
fl_server.py

Defines the Flower server logic for federated fine-tuning, including the aggregation strategy (FedAvg).
Sets up and runs the Flower simulation.
"""

import flwr as fl
import torch
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters, Scalar, EvaluateRes, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import logging
import numpy as np

# Assuming model.py, data_utils.py, fl_client.py are accessible
from .model import BitNetGainCellLLM, BitNetGainCellConfig
from .data_utils import load_mimic_data, preprocess_and_tokenize, partition_data
from .fl_client import client_fn, get_lora_config, get_peft_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Custom FedAvg Strategy (Optional, for potential modifications) ---
# We can use the default FedAvg, but a custom strategy allows for more control,
# e.g., logging, custom aggregation, or handling LoRA parameters specifically.

class FedAvgLora(FedAvg):
    """Custom FedAvg strategy to handle LoRA parameters and logging."""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate LoRA parameter updates using weighted average."""
        logging.info(f"Server: Aggregating fit results for round {server_round}")
        
        # Call the parent FedAvg aggregate_fit method
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            logging.info(f"Server: Aggregation successful for round {server_round}")
            # Convert aggregated NumPy arrays back to Parameters object
            # aggregated_parameters_obj = fl.common.ndarrays_to_parameters(aggregated_parameters)
        else:
            logging.warning(f"Server: Aggregation failed for round {server_round}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses."""
        logging.info(f"Server: Aggregating evaluation results for round {server_round}")
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        if loss_aggregated is not None:
            logging.info(f"Server: Round {server_round} aggregated evaluation loss: {loss_aggregated:.4f}")
        else:
             logging.warning(f"Server: Evaluation aggregation failed for round {server_round}")
        return loss_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        
        # Add standard FL parameters like learning rate, epochs for this round
        # These can be fixed or vary per round
        config["learning_rate"] = 1e-4 # Example LR
        config["local_epochs"] = 1      # Example local epochs
        config["current_round"] = server_round

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

# --- Main Simulation Function ---
def run_simulation(num_rounds=5, num_clients=16, batch_size=8, data_dir=None):
    """
    Sets up and runs the Flower simulation.

    Args:
        num_rounds (int): Number of federated learning rounds.
        num_clients (int): Total number of clients available for simulation.
        batch_size (int): Batch size for client dataloaders.
        data_dir (str, optional): Directory containing MIMIC data. Defaults to None, uses DEFAULT_DATA_DIR.
    """
    logging.info("--- Starting Federated Learning Simulation --- ")
    
    # 1. Load and Prepare Data
    if data_dir is None:
        data_dir = "/home/ubuntu/federated_llm/data" # Use default if not provided
    
    mimic_df = load_mimic_data(data_dir=data_dir)
    if mimic_df is None or mimic_df.empty:
        logging.error("Failed to load data. Aborting simulation.")
        return
    
    tokenized_data = preprocess_and_tokenize(mimic_df)
    client_datasets = partition_data(tokenized_data, num_clients=num_clients, alpha=0.5)
    logging.info(f"Data loaded, tokenized, and partitioned for {num_clients} clients.")

    # 2. Define Model Configuration
    # Use a smaller config for faster simulation if needed, otherwise use defaults
    model_config = BitNetGainCellConfig(
        block_size=512, # Match tokenizer max_length
        vocab_size=50257, # GPT-2 vocab size
        n_layer=6,     # Smaller model for simulation
        n_head=6,      
        n_embd=384,    
        dropout=0.1,
        bias=False
    )
    logging.info(f"Using model config: {model_config}")

    # 3. Define Federated Learning Strategy (FedAvg for LoRA)
    # We need initial parameters, but LoRA parameters are created *inside* the client.
    # For FedAvg with PEFT, the server often doesn't need initial PEFT params.
    # It sends global model params (which are frozen) and aggregates PEFT params.
    # However, Flower's FedAvg expects *some* initial parameters.
    # Let's provide dummy parameters initially, or modify the strategy.
    # A better approach: Initialize the strategy *without* initial parameters, 
    # and let the first round's parameters come from the clients.
    # Or, initialize a dummy PEFT model on the server just to get parameter shapes.
    
    # Dummy model to get initial parameter structure (only LoRA weights)
    temp_model = BitNetGainCellLLM(model_config)
    lora_conf = get_lora_config() # Defined in fl_client.py
    temp_peft_model = get_peft_model(temp_model, lora_conf)
    initial_lora_params = [val.cpu().numpy() for name, val in temp_peft_model.named_parameters() if "lora_" in name]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_lora_params)
    logging.info(f"Initialized strategy with {len(initial_lora_params)} dummy LoRA parameters.")
    del temp_model, temp_peft_model # Free memory

    strategy = FedAvgLora(
        fraction_fit=1.0,  # Sample 100% of clients for training each round
        fraction_evaluate=1.0, # Sample 100% for evaluation (can reduce if needed)
        min_fit_clients=num_clients, # Minimum clients for training
        min_evaluate_clients=num_clients, # Minimum clients for evaluation
        min_available_clients=num_clients, # Wait for all clients to be available
        initial_parameters=initial_parameters,
        # evaluate_fn=get_evaluate_fn(model_config, device), # Optional: Server-side evaluation
        # on_fit_config_fn=fit_config, # Optional: Function to configure client training each round
    )
    logging.info("Federated learning strategy initialized (FedAvgLora).")

    # 4. Define Client Resources
    # Determine device and resources per client for simulation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Simulation device: {device}")
    client_resources = None
    if device.type == "cuda":
        # Assign GPU resources if available. Adjust num_gpus based on your system.
        # Example: Allow each client to use 1 GPU if available
        # Flower simulation handles resource allocation.
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
             gpu_per_client = 1 # Assign 1 GPU per client if possible
             # Ensure we don't request more GPUs than available per client instance
             # This depends on how many clients run concurrently. Flower handles this.
             client_resources = {"num_cpus": 2, "num_gpus": gpu_per_client}
             logging.info(f"Assigning {gpu_per_client} GPU per client.")
        else:
             client_resources = {"num_cpus": 2, "num_gpus": 0}
             logging.warning("No GPUs detected by PyTorch. Running on CPU.")
    else:
        client_resources = {"num_cpus": 2, "num_gpus": 0}
        logging.info("Running on CPU.")

    # 5. Start Simulation
    logging.info(f"Starting Flower simulation with {num_clients} clients for {num_rounds} rounds.")
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, model_config, client_datasets, batch_size, device),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": "ray.util.accelerate.accelerate_torch_on_actor", # If using accelerate with Ray
        }
    )

    logging.info("--- Simulation Finished --- ")
    logging.info(f"History (losses_distributed): {history.losses_distributed}")
    # Add saving logic for history or final model if needed
    # np.save("/home/ubuntu/federated_llm/results/fl_history.npy", history)

    return history

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Run the simulation with specified parameters
    # Adjust parameters as needed
    run_simulation(
        num_rounds=3,      # Keep low for testing
        num_clients=4,     # Use fewer clients for faster testing
        batch_size=4,      # Smaller batch size for testing
        # data_dir="/path/to/your/mimic/csvs" # Optional: Specify data directory if not default
    )

