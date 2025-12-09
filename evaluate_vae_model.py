#!/usr/bin/env python3
"""
Evaluation script for VAE preference model on test data.
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader
import sys

# Add path for imports
sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

# Import VAE components
from vae_utils import VAEModel, VAETrainer

# Import classes we need from the training script (avoiding relative imports)
from transformers.utils import PaddingStrategy
import torch.nn as nn
from typing import Union, Optional


@dataclass
class ScriptArguments:
    """Arguments for the evaluation script."""
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_eval_batch_size: int = field(default=1)
    max_length: int = field(default=1024)
    fixed_contexts: bool = field(default=False)
    fixed_llm_embeddings: bool = field(default=False)
    other_subsets: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)


class RewardDataCollatorWithPadding:
    """Data collator for reward model training with padding."""

    def __init__(
        self,
        args: ScriptArguments,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of examples."""
        if self.args.other_subsets is None:
            user_mapping = {
                "helpful": 0,
                "harmless": 1,
            }
        else:
            if self.args.other_subsets == 'ultra_feedback':
                subsets = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
            elif self.args.other_subsets == 'single' or self.args.other_subsets == '84':
                subsets = ['8', '4', '2', '1']
            else:
                subsets = []
            user_mapping = {subset: idx for idx, subset in enumerate(subsets)}

        if self.args.fixed_llm_embeddings:
            batch_size = len(features)
            embeddings_chosen = []
            embeddings_rejected = []
            contexts_embeddings_chosen = []
            contexts_embeddings_rejected = []
            contexts_lengths = [0]

            for feature in features:
                embeddings_chosen.append(feature["embedding_chosen"])
                embeddings_rejected.append(feature["embedding_rejected"])
                contexts_embeddings_chosen.extend(
                    [ctx["embedding_chosen"] for ctx in feature["contexts_embeddings"]]
                )
                contexts_embeddings_rejected.extend(
                    [ctx["embedding_rejected"] for ctx in feature["contexts_embeddings"]]
                )
                contexts_lengths.append(len(feature["contexts_embeddings"]))

            contexts_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
            seq_start_end = torch.stack(
                [contexts_lengths[:-1], contexts_lengths[1:]], dim=1
            )
            user_type = [user_mapping.get(feature["data_subset"], 0) for feature in features]

            return {
                "embeddings_chosen": embeddings_chosen,
                "embeddings_rejected": embeddings_rejected,
                "contexts_embeddings_chosen": contexts_embeddings_chosen,
                "contexts_embeddings_rejected": contexts_embeddings_rejected,
                "seq_start_end": seq_start_end,
                "return_loss": True,
                "user_type": user_type,
            }
        return {}


def load_custom_dataset(data_path: str, split: str = "test") -> Dataset:
    """Load dataset from custom JSONL files in directory structure."""
    datasets: List[Dataset] = []

    # Check if directory structure exists
    base_path = Path(data_path)

    # Get subdirectories that contain test.jsonl files
    subdirs = sorted([d for d in base_path.glob('*/') if d.is_dir()])

    for subdir in subdirs:
        split_file = subdir / f"{split}.jsonl"
        if split_file.exists():
            print(f"Loading {split_file}")
            dataset = load_dataset('json', data_files=str(split_file), split=None)

            # Map data_subset based on directory name
            data_subset_name = subdir.name  # Use directory name as subset
            if isinstance(dataset, dict):
                # If load_dataset returns a DatasetDict
                dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]

            dataset = dataset.map(lambda x: {**x, "data_subset": data_subset_name})
            datasets.append(dataset)
            print(f"  Loaded {len(dataset)} examples from {subdir.name}")

    if not datasets:
        raise ValueError(f"No {split}.jsonl files found in {data_path}")

    combined_dataset = concatenate_datasets(datasets)
    print(f"Total samples loaded: {len(combined_dataset)}")
    return combined_dataset


def preprocess_dataset_with_embeddings(dataset: Dataset) -> Dataset:
    """Preprocess dataset to format embeddings for the model."""

    def format_embeddings(example):
        """Convert embedding lists to tensors."""
        return {
            "embedding_chosen": example["embeddings"]["embedding_chosen"],
            "embedding_rejected": example["embeddings"]["embedding_rejected"],
            "contexts_embeddings": [
                {
                    "embedding_chosen": ctx["embedding_chosen"],
                    "embedding_rejected": ctx["embedding_rejected"]
                }
                for ctx in example["contexts"]
            ],
            "max_lengths": 0,  # Placeholder
            "data_subset": example["data_subset"],
        }

    # Apply the formatting
    dataset = dataset.map(
        format_embeddings,
        remove_columns=[col for col in dataset.column_names
                       if col not in ["data_subset"]],
    )

    return dataset


def load_checkpoint(
    checkpoint_path: str,
    model_name: str = "gpt2",
    encoder_embed_dim: int = 768,
    decoder_embed_dim: int = 768,
    hidden_dim: int = 512,
    latent_dim: int = 512,
) -> tuple:
    """Load the VAE model from checkpoint."""
    from peft import get_peft_model, LoraConfig, TaskType

    # Load base models
    llm_encoder = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=decoder_embed_dim, torch_dtype=torch.bfloat16
    )
    llm_encoder.score.weight.data *= 0.01

    contexts_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=encoder_embed_dim, torch_dtype=torch.bfloat16
    )
    contexts_model.score.weight.data *= 0.01

    # Apply PEFT adapters if they exist
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")

    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        print("Loading PEFT adapters from checkpoint...")
        llm_encoder = PeftModel.from_pretrained(llm_encoder, checkpoint_path)
        contexts_model = PeftModel.from_pretrained(contexts_model, checkpoint_path)
    else:
        # If no adapters, apply PEFT config
        print("No PEFT adapters found. Using base models.")

    # Setup tokenizer settings
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_encoder.config.pad_token_id = tokenizer.pad_token_id
    contexts_model.config.pad_token_id = tokenizer.pad_token_id

    # Load VAE model
    model_pt_path = os.path.join(checkpoint_path, "model.pt")
    if os.path.exists(model_pt_path):
        print(f"Loading VAE model from {model_pt_path}")
        vae_model = torch.load(model_pt_path, map_location='cpu', weights_only=False)
    else:
        # Create VAE model from scratch if model.pt doesn't exist
        print(f"Warning: model.pt not found at {model_pt_path}")
        print("Creating VAE model with base encoders...")
        vae_model = VAEModel(
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            llm_encoder=llm_encoder,
            llm_contexts_encoder=contexts_model,
            fixed_contexts=False,
            fixed_llm_embeddings=False,
        )

    vae_model.eval()
    return vae_model, tokenizer


def evaluate_model(
    model: VAEModel,
    test_dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Evaluate the model on test data."""
    model = model.to(device)
    model.eval()

    all_rewards_chosen = []
    all_rewards_rejected = []
    all_means = []
    all_log_vars = []
    all_z = []
    all_user_types = []

    losses = []
    accuracies = []

    print(f"Evaluating on {device}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}...")

            # Move batch to device - handle both tensor and list types
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    batch_on_device[k] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
                else:
                    batch_on_device[k] = v

            # Forward pass through the model
            try:
                embeddings_chosen = torch.stack([torch.tensor(e, dtype=torch.bfloat16) for e in batch_on_device["embeddings_chosen"]]).to(device)
                embeddings_rejected = torch.stack([torch.tensor(e, dtype=torch.bfloat16) for e in batch_on_device["embeddings_rejected"]]).to(device)

                # Stack context embeddings
                contexts_embeddings_chosen = torch.stack([torch.tensor(e, dtype=torch.bfloat16) for e in batch_on_device["contexts_embeddings_chosen"]]).to(device)
                contexts_embeddings_rejected = torch.stack([torch.tensor(e, dtype=torch.bfloat16) for e in batch_on_device["contexts_embeddings_rejected"]]).to(device)

                seq_start_end = batch_on_device["seq_start_end"].to(device)
                user_type_batch = torch.tensor(batch_on_device["user_type"], dtype=torch.bfloat16).to(device)

                # Forward pass through VAE model
                rewards_chosen, rewards_rejected, mean, log_var, z = model(
                    target_chosen=embeddings_chosen,
                    target_rejected=embeddings_rejected,
                    context_chosen=contexts_embeddings_chosen,
                    context_rejected=contexts_embeddings_rejected,
                    seq_start_end=seq_start_end,
                    user_type=user_type_batch,
                    ground_truth_user_vector=False,
                )
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Prepare outputs for metric computation
            user_type = batch_on_device["user_type"]

            # Store results (convert bfloat16 to float32 for numpy compatibility)
            all_rewards_chosen.append(rewards_chosen.float().cpu().numpy())
            all_rewards_rejected.append(rewards_rejected.float().cpu().numpy())
            all_means.append(mean.float().cpu().numpy())
            all_log_vars.append(log_var.float().cpu().numpy())
            all_z.append(z.float().cpu().numpy())
            all_user_types.extend(user_type.cpu().numpy().tolist() if isinstance(user_type, torch.Tensor) else user_type)

            # Calculate metrics
            per_sample_loss = -torch.nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected
            )
            loss_val = per_sample_loss.mean().item()
            accuracy = (rewards_chosen > rewards_rejected).float().mean().item()

            losses.append(loss_val)
            accuracies.append(accuracy)

    # Aggregate results
    all_rewards_chosen = np.concatenate(all_rewards_chosen, axis=0)
    all_rewards_rejected = np.concatenate(all_rewards_rejected, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    all_log_vars = np.concatenate(all_log_vars, axis=0)
    all_z = np.concatenate(all_z, axis=0)

    # Compute metrics
    per_sample_loss = -np.log(1 / (1 + np.exp(-(all_rewards_chosen - all_rewards_rejected))))

    # KLD computation
    kld = -0.5 * np.sum(1 + all_log_vars - all_means**2 - np.exp(all_log_vars), axis=1)

    metrics = {
        "num_samples": len(all_rewards_chosen),
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(accuracies)),
        "kld_mean": float(np.mean(kld)),
        "kld_std": float(np.std(kld)),
        "rewards_chosen_mean": float(np.mean(all_rewards_chosen)),
        "rewards_chosen_std": float(np.std(all_rewards_chosen)),
        "rewards_rejected_mean": float(np.mean(all_rewards_rejected)),
        "rewards_rejected_std": float(np.std(all_rewards_rejected)),
        "mean_embeddings_norm": float(np.mean(np.linalg.norm(all_means, axis=1))),
        "z_embeddings_norm": float(np.mean(np.linalg.norm(all_z, axis=1))),
    }

    return metrics, {
        "rewards_chosen": all_rewards_chosen,
        "rewards_rejected": all_rewards_rejected,
        "means": all_means,
        "log_vars": all_log_vars,
        "z": all_z,
        "user_types": all_user_types,
    }


def main():
    # Configuration - adjust these based on your checkpoint
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"
    test_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    output_dir = "/hpc/group/fanglab/xx102/vpl_llm/evaluation_results"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup arguments - using fixed embeddings since data has pre-computed embeddings
    script_args = ScriptArguments()
    script_args.max_length = 1024
    script_args.per_device_eval_batch_size = 1
    script_args.fixed_contexts = True
    script_args.fixed_llm_embeddings = True

    # Load checkpoint
    print("\n=== Loading Checkpoint ===")
    vae_model, tokenizer = load_checkpoint(
        checkpoint_path,
        model_name="gpt2",
        encoder_embed_dim=768,
        decoder_embed_dim=768,
        hidden_dim=512,
        latent_dim=512,
    )

    # Load test data
    print("\n=== Loading Test Data ===")
    test_dataset = load_custom_dataset(test_data_path, split="test")
    print(f"Loaded {len(test_dataset)} test samples")

    # Preprocess dataset with embeddings
    print("\n=== Preprocessing Dataset ===")
    test_dataset = preprocess_dataset_with_embeddings(test_dataset)
    print(f"After preprocessing: {len(test_dataset)} samples")

    # Create data loader
    print("\n=== Creating DataLoader ===")
    data_collator = RewardDataCollatorWithPadding(
        args=script_args,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        pad_to_multiple_of=64,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=script_args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    # Evaluate
    print("\n=== Evaluating Model ===")
    metrics, predictions = evaluate_model(vae_model, test_dataloader, device=device)

    # Print results
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save results
    print(f"\n=== Saving Results to {output_dir} ===")

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save predictions
    predictions_path = os.path.join(output_dir, "predictions.npz")
    np.savez(predictions_path, **predictions)
    print(f"Predictions saved to {predictions_path}")

    # Create summary report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("VAE Preference Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Data: {test_data_path}\n")
        f.write(f"Test Samples: {metrics['num_samples']}\n\n")
        f.write("Metrics:\n")
        f.write("-" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key:.<40} {value:.6f}\n")
    print(f"Report saved to {report_path}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
