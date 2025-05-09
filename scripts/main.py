# -*- coding: utf-8 -*-
"""
main.py

Main script to run the federated fine-tuning simulation.
"""

import argparse
import logging
import os

# Assuming fl_server.py is in the src directory relative to scripts
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "E:\Memristorproject\FLLM")) # Add project root to path

from src .fl_server import run_simulation

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run Federated LLM Fine-tuning Simulation")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated learning rounds.")
    parser.add_argument("--num_clients", type=int, default=16, help="Total number of clients to simulate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for client dataloaders.")
    parser.add_argument("--data_dir", type=str, default="E:\Memristorproject\FLLM\data", help="Directory containing MIMIC-IV-Ext-CDM CSV files.")
    # Add other relevant arguments if needed (e.g., learning rate, local epochs)

    args = parser.parse_args()

    logging.info("Starting main simulation script...")
    logging.info(f"Arguments: Rounds={args.num_rounds}, Clients={args.num_clients}, BatchSize={args.batch_size}, DataDir={args.data_dir}")

    # Check if data directory exists
    if not os.path.isdir(args.data_dir):
        logging.error(f"Data directory not found: {args.data_dir}")
        logging.error("Please ensure the MIMIC-IV-Ext-CDM CSV files are placed in this directory.")
        logging.error("Aborting simulation.")
        return
    elif not os.listdir(args.data_dir):
        logging.warning(f"Data directory {args.data_dir} is empty.")
        logging.warning("Attempting to run simulation, but data loading will likely fail.")
        logging.warning("Please ensure the MIMIC-IV-Ext-CDM CSV files are placed in this directory.")

    # Run the simulation
    try:
        run_simulation(
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            data_dir=args.data_dir
        )
        logging.info("Simulation run completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}", exc_info=True)
        logging.error("Simulation aborted due to error.")

if __name__ == "__main__":
    main()

