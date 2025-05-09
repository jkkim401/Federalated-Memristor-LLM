import flwr as fl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import PeftModel, PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
from collections import OrderedDict
import logging

from .model import BitNetGainCellLLM, BitNetGainCellConfig
# from .data_utils import load_mimic_data, preprocess_and_tokenize # Data loading happens in main script

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- LoRA Configuration ---
def get_lora_config():
    """Returns the LoRA configuration for the BitNet model."""
    # Configure LoRA. Target modules might need adjustment based on the exact BitNet/GainCell structure.
    # Common targets are query, key, value projections in attention and linear layers in MLP.
    # Since we use BitLinear, we target those.
    # Note: BitNet's quantization might interact with LoRA; this needs careful consideration/testing.
    # We apply LoRA to the original nn.Linear layers *before* they are potentially replaced or wrapped
    # by BitLinear, OR we need to ensure LoRA can wrap BitLinear correctly.
    # Let's assume LoRA wraps the BitLinear layers directly if PEFT supports it, or target internal weights.
    # For simplicity, targeting names assuming they appear as standard nn.Linear or compatible.
    # Adjust target_modules based on `print(model)` output.
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32, # LoRA alpha scaling
        # target_modules=["query", "key", "value", "proj", "mlp.0", "mlp.3"], # Adjust based on model structure
        # Targeting BitLinear layers directly might require PEFT library support or custom wrapping.
        # Let's assume we target the names within the Block structure:
        target_modules=["attn.query.weight", "attn.key.weight", "attn.value.weight", "attn.proj.weight", "mlp.0.weight", "mlp.3.weight"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM # Assuming a causal language modeling task
    )
    return lora_config

# --- Flower Client Definition ---
class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated LoRA fine-tuning."""
    def __init__(self, cid, model_config, trainloader, valloader, device):
        """
        Initializes the Flower client.

        Args:
            cid (str): Client ID.
            model_config (BitNetGainCellConfig): Configuration for the model.
            trainloader (torch.utils.data.DataLoader): DataLoader for the client's training data.
            valloader (torch.utils.data.DataLoader): DataLoader for the client's validation data.
            device (torch.device): Device to run training on (e.g., CUDA, CPU).
        """
        model_name = "decapoda-research/llama-3.1-8b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        base_llama = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

        bitnet_model = BitNetGainCellLLM(model_config)

        hf_state_dict = base_llama.state_dict()
        bitnet_state_dict = bitnet_model.state_dict()

        for key in hf_state_dict:
            if key in bitnet_state_dict:
                bitnet_state_dict[key] = hf_state_dict[key]
                
        bitnet_model.load_state_dict(bitnet_state_dict)

        lora_config = get_lora_config(r=16, lora_alpha=32, 
                                      target_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", 
                                                      "mlp.fc1", "mlp.fc2"], 
                                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(bitnet_model, lora_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()

        self.cid = cid
        self.model_config = model_config
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

        # Instantiate the base model
        self.base_model = BitNetGainCellLLM(self.model_config).to(self.device)
        # Apply LoRA configuration
        self.lora_config = get_lora_config()
        self.model = get_peft_model(self.base_model, self.lora_config)
        logging.info(f"Client {self.cid}: Model loaded and LoRA applied. Trainable params:")
        self.model.print_trainable_parameters()

    def get_parameters(self, config):
        """Return the LoRA parameters of the model."""
        # Return only the trainable LoRA parameters
        lora_params = [val.cpu().numpy() for name, val in self.model.named_parameters() if "lora_" in name]
        logging.info(f"Client {self.cid}: Getting {len(lora_params)} LoRA parameters.")
        return lora_params

    def set_parameters(self, parameters):
        """Update the model's LoRA parameters."""
        # Load the received LoRA parameters
        lora_param_names = [name for name, _ in self.model.named_parameters() if "lora_" in name]
        if len(parameters) != len(lora_param_names):
             logging.error(f"Client {self.cid}: Mismatch in number of parameters received ({len(parameters)}) vs expected ({len(lora_param_names)}).")
             return

        params_dict = zip(lora_param_names, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        try:
            incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                 logging.warning(f"Client {self.cid}: Incompatible keys when loading state dict: {incompatible_keys}")
            else:
                 logging.info(f"Client {self.cid}: Successfully set {len(parameters)} LoRA parameters.")
        except Exception as e:
            logging.error(f"Client {self.cid}: Error loading state dict: {e}")


    def fit(self, parameters, config):
        """Train the model using the client's local data."""
        logging.info(f"Client {self.cid}: Starting local training (fit). Rounds: {config.get("local_epochs", 1)}")
        self.set_parameters(parameters) # Update model with parameters from server

        # --- Training Loop ---
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=float(config.get("learning_rate", 5e-5)))
        # Simple scheduler, adjust as needed
        num_epochs = int(config.get("local_epochs", 1))
        num_training_steps = num_epochs * len(self.trainloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self.trainloader:
                optimizer.zero_grad()
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Assuming model takes input_ids and returns loss when labels are provided
                outputs = self.model(idx=batch["input_ids"], targets=batch["input_ids"]) # Use input_ids as targets for Causal LM
                loss = outputs[1] # model returns (logits, loss)
                
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                else:
                    logging.warning(f"Client {self.cid}: Loss is None for a batch.")

            avg_loss = total_loss / len(self.trainloader) if len(self.trainloader) > 0 else 0
            logging.info(f"Client {self.cid}: Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Return updated LoRA parameters and number of training examples
        updated_params = self.get_parameters(config={})
        return updated_params, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
        logging.info(f"Client {self.cid}: Starting local evaluation (evaluate).")
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0
        num_examples = 0
        with torch.no_grad():
            for batch in self.valloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(idx=batch["input_ids"], targets=batch["input_ids"])
                loss = outputs[1]
                if loss is not None:
                    total_loss += loss.item() * batch["input_ids"].size(0) # Weighted by batch size
                    num_examples += batch["input_ids"].size(0)
                else:
                     logging.warning(f"Client {self.cid}: Loss is None during evaluation.")

        avg_loss = total_loss / num_examples if num_examples > 0 else 0
        logging.info(f"Client {self.cid}: Evaluation Loss: {avg_loss:.4f}")

        # Return evaluation loss, number of examples, and potentially other metrics
        return float(avg_loss), num_examples, {"loss": float(avg_loss)}

# --- Helper function to create clients ---
def client_fn(cid: str, model_config, all_client_datasets, batch_size, device):
    """Create a Flower client instance for simulation."""
    client_dataset = all_client_datasets[int(cid)]
    # Simple split for train/val, adjust ratio if needed
    train_size = int(0.9 * len(client_dataset))
    val_size = len(client_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(client_dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    return FlowerClient(cid, model_config, trainloader, valloader, device)

# Example Usage (Conceptual - called from main script)
if __name__ == '__main__':
    # This part is conceptual and would be run from a main simulation script
    logging.info("--- Running FL Client Example (Conceptual) ---")
    
    # 1. Define Model Config
    config = BitNetGainCellConfig(
        block_size=64, vocab_size=500, n_layer=2, n_head=4, n_embd=128, dropout=0.1, bias=False
    )
    
    # 2. Prepare Dummy Data (Replace with actual data loading)
    from datasets import Dataset
    dummy_data = {"input_ids": torch.randint(0, config.vocab_size, (100, config.block_size))}
    full_dataset = Dataset.from_dict(dummy_data)
    full_dataset.set_format("torch")
    client_datasets = [full_dataset.select(range(i*10, (i+1)*10)) for i in range(10)] # Dummy partition

    # 3. Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 4. Instantiate a client (example for client '0')
    client = client_fn("0", config, client_datasets, batch_size=4, device=device)
    logging.info("Client instance created.")

    # 5. Simulate get/set parameters and fit/evaluate (conceptual)
    initial_params = client.get_parameters(config={})
    logging.info(f"Got {len(initial_params)} initial LoRA parameters.")
    
    # Simulate a fit round
    fit_config = {"local_epochs": 1, "learning_rate": 1e-4}
    updated_params, num_examples_train, metrics_train = client.fit(initial_params, fit_config)
    logging.info(f"Fit completed. Trained on {num_examples_train} examples. Got {len(updated_params)} updated parameters.")

    # Simulate an evaluate round
    eval_loss, num_examples_eval, metrics_eval = client.evaluate(updated_params, config={})
    logging.info(f"Evaluate completed. Loss: {eval_loss:.4f} on {num_examples_eval} examples.")

    logging.info("--- FL Client Example Finished ---")

