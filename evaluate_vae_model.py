#!/usr/bin/env python3
"""
Evaluation script for VAE preference model on test data.
"""
import os
import json
import torch
import numpy as np
import argparse
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

# Import dataset utilities
from dataset_utils import load_and_split_survey_data

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
    controversial_only: bool = field(default=False)


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
            elif self.args.other_subsets:
                subsets = [self.args.other_subsets]
            else:
                subsets = []
            user_mapping = {subset: idx for idx, subset in enumerate(subsets)}

        if self.args.fixed_llm_embeddings:
            # Both target and context embeddings are fixed
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
        elif self.args.fixed_contexts:
            # Target embeddings computed from text, context embeddings fixed
            batch_size = len(features)
            features_chosen = []
            features_rejected = []
            contexts_embeddings_chosen = []
            contexts_embeddings_rejected = []
            contexts_lengths = [0]

            for feature in features:
                features_chosen.append({
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                })
                features_rejected.append({
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                })
                contexts_embeddings_chosen.extend(
                    [ctx["embedding_chosen"] for ctx in feature["contexts_embeddings"]]
                )
                contexts_embeddings_rejected.extend(
                    [ctx["embedding_rejected"] for ctx in feature["contexts_embeddings"]]
                )
                contexts_lengths.append(len(feature["contexts_embeddings"]))

            # Tokenize padding
            batch = self.tokenizer.pad(
                features_chosen + features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            input_ids = batch["input_ids"].view(2, batch_size, batch["input_ids"].shape[-1])
            attention_mask = batch["attention_mask"].view(2, batch_size, batch["attention_mask"].shape[-1])

            context_lengths = torch.cumsum(torch.tensor(contexts_lengths), dim=0)
            seq_start_end = torch.stack(
                [context_lengths[:-1], context_lengths[1:]], dim=1
            )
            user_type = [user_mapping.get(feature["user_type"], 0) for feature in features]
            assert len(seq_start_end) == batch_size

            return {
                "input_ids_chosen": input_ids[0],
                "attention_mask_chosen": attention_mask[0],
                "input_ids_rejected": input_ids[1],
                "attention_mask_rejected": attention_mask[1],
                "contexts_embeddings_chosen": contexts_embeddings_chosen,
                "contexts_embeddings_rejected": contexts_embeddings_rejected,
                "seq_start_end": seq_start_end,
                "return_loss": True,
                "user_type": user_type,
            }

        return {}


class SurveyContextPreprocessor:
    """Data preprocessor for evaluation with randomly sampled survey contexts."""

    def __init__(
        self,
        args: ScriptArguments,
        tokenizer: PreTrainedTokenizerBase,
        demo_datasets: Dict[str, Dataset],
        num_contexts: int = 8,
        random_seed: int = 0,
        **tokenizer_kwargs
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer_kwargs = tokenizer_kwargs
        self.demo_datasets = demo_datasets
        self.num_contexts = num_contexts
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.example_counter = 0

    def _sample_contexts(self, subset: str, seed_offset: int = 0) -> List[Dict[str, Any]]:
        """Sample random contexts from demo dataset for a given subset."""
        demo_dataset = self.demo_datasets[subset]

        # Ensure we have enough demo samples
        if len(demo_dataset) < self.num_contexts:
            raise ValueError(
                f"Subset {subset}: demo dataset has {len(demo_dataset)} samples "
                f"but need {self.num_contexts} for context sampling"
            )

        # Per-example deterministic sampling
        rng = np.random.RandomState(self.random_seed + seed_offset)
        sampled_indices = rng.choice(len(demo_dataset), self.num_contexts, replace=False)

        contexts = []
        for idx in sampled_indices:
            example = demo_dataset[int(idx)]
            contexts.append({
                "embedding_chosen": example["embeddings"]["embedding_chosen"],
                "embedding_rejected": example["embeddings"]["embedding_rejected"],
            })

        return contexts

    def __call__(self, examples):
        if self.args.fixed_llm_embeddings:
            # Fixed embeddings mode
            new_examples = {
                "embedding_chosen": [],
                "embedding_rejected": [],
                "contexts_embeddings": [],
                "max_lengths": []
            }
            for i, (embeddings, data_subset) in enumerate(
                zip(examples["embeddings"], examples["data_subset"])
            ):
                new_examples["embedding_chosen"].append(embeddings["embedding_chosen"])
                new_examples["embedding_rejected"].append(embeddings["embedding_rejected"])

                # Sample random contexts from survey data
                contexts_embeddings = self._sample_contexts(data_subset, seed_offset=self.example_counter + i)
                new_examples["contexts_embeddings"].append(contexts_embeddings)
                new_examples["max_lengths"].append(0)

            new_examples["user_type"] = examples["data_subset"]
            self.example_counter += len(examples["embeddings"])
            return new_examples
        else:
            # Tokenize chosen/rejected but use randomly sampled context embeddings
            new_examples = {
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": [],
                "contexts_embeddings": [],
                "max_lengths": []
            }

            for i, (chosen, rejected, data_subset) in enumerate(
                zip(examples["chosen"], examples["rejected"], examples["data_subset"])
            ):
                max_length = 0
                tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
                tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

                new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
                new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
                new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
                new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

                max_length = max(max_length, len(tokenized_chosen["input_ids"]))
                max_length = max(max_length, len(tokenized_rejected["input_ids"]))

                # Sample random contexts from survey data
                contexts_embeddings = self._sample_contexts(data_subset, seed_offset=self.example_counter + i)
                new_examples["contexts_embeddings"].append(contexts_embeddings)
                new_examples["max_lengths"].append(max_length)

            new_examples["user_type"] = examples["data_subset"]
            self.example_counter += len(examples["chosen"])
            return new_examples




def load_custom_dataset(data_path: str, split: str = "test", subset = "8") -> Dataset:
    """Load dataset from custom JSONL files in directory structure.

    Args:
        data_path: Path to data directory containing subset subdirectories
        split: Which split to load (e.g., 'test', 'train')
        subset: Which subset(s) to load. Can be a string (e.g., '8') or list of strings (e.g., ['8', '4', '2', '1']).
                Default: '8'
    """
    base_path = Path(data_path)

    # Handle both single subset (string) and multiple subsets (list)
    if isinstance(subset, str):
        subsets = [subset]
    else:
        subsets = subset

    datasets = []

    for sub in subsets:
        subset_dir = base_path / sub
        split_file = subset_dir / f"{split}.jsonl"

        if not split_file.exists():
            raise ValueError(f"Dataset file not found: {split_file}")

        print(f"Loading {split_file}")
        dataset = load_dataset('json', data_files=str(split_file), split=None)

        # Map data_subset based on directory name
        data_subset_name = subset_dir.name
        if isinstance(dataset, dict):
            # If load_dataset returns a DatasetDict
            dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]

        dataset = dataset.map(lambda x: {**x, "data_subset": data_subset_name})
        print(f"Loaded {len(dataset)} examples from subset {sub}")
        datasets.append(dataset)

    # Concatenate all datasets if multiple subsets
    if len(datasets) > 1:
        combined_dataset = concatenate_datasets(datasets)
        print(f"Total: {len(combined_dataset)} examples from {len(datasets)} subsets")
        return combined_dataset
    else:
        return datasets[0]


def preprocess_dataset_matching_training(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    use_survey_contexts: bool = False,
    demo_datasets: Optional[Dict[str, Dataset]] = None,
    num_contexts: int = 8,
    random_seed: int = 0,
    num_proc: int = 24
) -> Dataset:
    """Preprocess dataset to match training configuration.

    Args:
        dataset: The test dataset to preprocess
        tokenizer: Tokenizer for encoding text
        args: Script arguments
        use_survey_contexts: If True, use SurveyContextPreprocessor for random context sampling
        demo_datasets: Demo datasets for context sampling (required if use_survey_contexts=True)
        num_contexts: Number of contexts to sample per example (default: 8)
        random_seed: Random seed for reproducibility (default: 0)
        num_proc: Number of processes for parallel processing (default: 24)

    Returns:
        Preprocessed dataset
    """
    # Use survey contexts with random sampling
    if use_survey_contexts:
        if demo_datasets is None:
            raise ValueError("demo_datasets required when use_survey_contexts=True")

        original_columns = dataset.column_names
        preprocessor = SurveyContextPreprocessor(
            args,
            tokenizer,
            demo_datasets=demo_datasets,
            num_contexts=num_contexts,
            random_seed=random_seed,
            truncation=True,
            max_length=args.max_length
        )

        dataset = dataset.map(
            preprocessor,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )

        # Filter by max length
        dataset = dataset.filter(lambda x: x["max_lengths"] <= args.max_length)

        return dataset

    # Original behavior with pre-computed contexts from test.jsonl
    from vae_utils import VAEModel  # Import here to get access to HHRLHFPreprocessor

    # Import HHRLHFPreprocessor from training script
    # We'll define it inline to match the training behavior exactly
    class EvalHHRLHFPreprocessor:
        def __init__(self, args, tokenizer, **tokenizer_kwargs):
            self.tokenizer = tokenizer
            self.args = args
            self.tokenizer_kwargs = tokenizer_kwargs

        def __call__(self, examples):
            if self.args.fixed_llm_embeddings:
                # Fixed embeddings mode
                new_examples = {
                    "embedding_chosen": [],
                    "embedding_rejected": [],
                    "contexts_embeddings": [],
                    "max_lengths": []
                }
                for embeddings, contexts in zip(
                    examples["embeddings"], examples["contexts"]
                ):
                    new_examples["embedding_chosen"].append(embeddings["embedding_chosen"])
                    new_examples["embedding_rejected"].append(embeddings["embedding_rejected"])
                    contexts_embeddings = [
                        {
                            "embedding_chosen": context["embedding_chosen"],
                            "embedding_rejected": context["embedding_rejected"]
                        }
                        for context in contexts
                    ]
                    new_examples["contexts_embeddings"].append(contexts_embeddings)
                    new_examples["max_lengths"].append(0)
                new_examples["user_type"] = examples["data_subset"]
                return new_examples
            else:
                # Tokenize chosen/rejected but use pre-computed context embeddings
                new_examples = {
                    "input_ids_chosen": [],
                    "attention_mask_chosen": [],
                    "input_ids_rejected": [],
                    "attention_mask_rejected": [],
                    "contexts_embeddings": [],
                    "max_lengths": []
                }
                for chosen, rejected, contexts, user_type in zip(
                    examples["chosen"], examples["rejected"], examples["contexts"], examples["data_subset"]
                ):
                    max_length = 0
                    tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
                    tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

                    new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
                    new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
                    new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
                    new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

                    max_length = max(max_length, len(tokenized_chosen["input_ids"]))
                    max_length = max(max_length, len(tokenized_rejected["input_ids"]))

                    # Use pre-computed context embeddings
                    contexts_embeddings = [
                        {
                            "embedding_chosen": context["embedding_chosen"],
                            "embedding_rejected": context["embedding_rejected"]
                        }
                        for context in contexts
                    ]
                    new_examples["contexts_embeddings"].append(contexts_embeddings)
                    new_examples["max_lengths"].append(max_length)

                new_examples["user_type"] = examples["data_subset"]
                return new_examples

    original_columns = dataset.column_names

    # Apply preprocessor
    dataset = dataset.map(
        EvalHHRLHFPreprocessor(args, tokenizer, truncation=True, max_length=args.max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    # Filter by max length
    dataset = dataset.filter(lambda x: x["max_lengths"] <= args.max_length)

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
        try:
            llm_encoder = PeftModel.from_pretrained(llm_encoder, checkpoint_path)
            llm_encoder.eval()  # Set to eval mode immediately
            contexts_model = PeftModel.from_pretrained(contexts_model, checkpoint_path)
            contexts_model.eval()  # Set to eval mode immediately
        except Exception as e:
            print(f"Warning: Error loading PEFT adapters: {e}")
            print("Continuing with base models...")
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

        # Update the VAE model's internal encoders with the ones that have PEFT adapters
        vae_model.llm_encoder = llm_encoder
        vae_model.llm_contexts_encoder = contexts_model
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

            # Forward pass through the model (matching training code structure)
            try:
                seq_start_end = batch_on_device["seq_start_end"].to(device)
                user_type_batch = torch.tensor(batch_on_device["user_type"], dtype=torch.bfloat16).to(device)

                # Handle fixed_llm_embeddings case
                if model.fixed_llm_embeddings:
                    # Pre-computed target embeddings
                    embeddings_chosen = torch.tensor(batch_on_device["embeddings_chosen"], dtype=torch.bfloat16).to(device)
                    embeddings_rejected = torch.tensor(batch_on_device["embeddings_rejected"], dtype=torch.bfloat16).to(device)
                else:
                    # Compute embeddings from tokenized inputs
                    output = model.llm_encoder(
                        input_ids=torch.concatenate([
                            batch_on_device["input_ids_chosen"],
                            batch_on_device["input_ids_rejected"],
                        ], dim=0),
                        attention_mask=torch.concatenate([
                            batch_on_device["attention_mask_chosen"],
                            batch_on_device["attention_mask_rejected"],
                        ], dim=0),
                    )
                    embeddings = output[0]
                    batch_size = batch_on_device["seq_start_end"].shape[0]
                    embeddings_chosen = embeddings[:batch_size]
                    embeddings_rejected = embeddings[batch_size:]

                # Stack context embeddings into tensors
                contexts_embeddings_chosen = torch.tensor(batch_on_device["contexts_embeddings_chosen"], dtype=torch.bfloat16).to(device)
                contexts_embeddings_rejected = torch.tensor(batch_on_device["contexts_embeddings_rejected"], dtype=torch.bfloat16).to(device)

                # Debug: Print shapes and sample values on first batch
                if batch_idx == 0:
                    print(f"\n  Debug Info (Batch 0):")
                    print(f"    embeddings_chosen shape: {embeddings_chosen.shape}, mean: {embeddings_chosen.mean():.4f}, std: {embeddings_chosen.std():.4f}")
                    print(f"    embeddings_rejected shape: {embeddings_rejected.shape}, mean: {embeddings_rejected.mean():.4f}, std: {embeddings_rejected.std():.4f}")
                    print(f"    contexts_chosen shape: {contexts_embeddings_chosen.shape}")
                    print(f"    contexts_rejected shape: {contexts_embeddings_rejected.shape}")
                    print(f"    seq_start_end: {seq_start_end}")
                    print(f"    embeddings_chosen vs rejected difference (norm): {(embeddings_chosen - embeddings_rejected).norm():.6f}")

                # Forward pass through VAE model (using positional args like training code)
                rewards_chosen, rewards_rejected, mean, log_var, z = model(
                    embeddings_chosen,
                    embeddings_rejected,
                    contexts_embeddings_chosen,
                    contexts_embeddings_rejected,
                    seq_start_end,
                    user_type_batch,
                    False,  # ground_truth_user_vector
                )

                # Debug: Print reward statistics
                if batch_idx == 0:
                    print(f"    rewards_chosen: {rewards_chosen.flatten()[:5]}, mean: {rewards_chosen.mean():.4f}")
                    print(f"    rewards_rejected: {rewards_rejected.flatten()[:5]}, mean: {rewards_rejected.mean():.4f}")
                    print(f"    mean (user vector): norm={mean.norm():.4f}, values={mean[0, :3]}")
                    print(f"    z (sampled user vector): norm={z.norm():.4f}, values={z[0, :3]}")
                    print()
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate VAE preference model on test data")
    parser.add_argument(
        "--subset",
        type=str,
        default="8",
        choices=["1", "2", "4", "8", "single", "84"],
        help="Data subset to evaluate: '8', '4', '2', '1', 'single' (8,4,2,1), or '84' (8,4). Default: 8"
    )
    parser.add_argument(
        "--use_survey_contexts",
        action="store_true",
        default=False,
        help="Use randomly sampled contexts from survey_100.jsonl instead of pre-computed contexts from test.jsonl"
    )
    parser.add_argument(
        "--num_contexts",
        type=int,
        default=8,
        help="Number of contexts to sample per test example (default: 8)"
    )
    parser.add_argument(
        "--context_seed",
        type=int,
        default=0,
        help="Random seed for context sampling (default: 0)"
    )
    args = parser.parse_args()

    # Configuration - adjust these based on your checkpoint
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"
    test_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    output_dir = f"/hpc/group/fanglab/xx102/vpl_llm/evaluation_results_subset_{args.subset}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Evaluating subset: {args.subset}")

    # Setup arguments - matching training script arguments from submit_job_UF_P_4.sh
    script_args = ScriptArguments()
    script_args.max_length = 1024
    script_args.per_device_eval_batch_size = 1
    script_args.fixed_contexts = True  # Use pre-computed context embeddings
    script_args.fixed_llm_embeddings = False  # Compute target embeddings from text via LLM encoder
    script_args.other_subsets = args.subset  # Use specified subset
    script_args.controversial_only = True  # Only evaluate on controversial examples

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
    # Determine which subsets to load based on other_subsets setting
    if script_args.other_subsets == 'ultra_feedback':
        subsets = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
    elif script_args.other_subsets == 'single':
        subsets = ['8', '4', '2', '1']
    elif script_args.other_subsets == '84':
        subsets = ['8', '4']
    elif script_args.other_subsets:
        subsets = [script_args.other_subsets]
    else:
        subsets = ['helpful', 'harmless']

    test_dataset = load_custom_dataset(test_data_path, split="test", subset=subsets)
    print(f"Loaded {len(test_dataset)} test samples")

    # Filter for controversial only (matching training script)
    if script_args.controversial_only:
        print("\n=== Filtering to Controversial Examples ===")
        test_dataset = test_dataset.filter(lambda example: example.get('controversial', False) == True)
        print(f"After filtering to controversial: {len(test_dataset)} samples")

    # Load and split survey data for context sampling if requested
    demo_datasets = None
    if args.use_survey_contexts:
        print("\n=== Loading Survey Data for Context Sampling ===")
        survey_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/UltraFeedback_single_P_4"
        demo_datasets, val_datasets = load_and_split_survey_data(
            survey_data_path,
            subsets=subsets,
            demo_size=50,
            validation_size=50,
            seed=args.context_seed,
        )
        print(f"Survey data split complete. Using demo sets for context sampling.")

    # Preprocess dataset matching training configuration
    print("\n=== Preprocessing Dataset ===")
    test_dataset = preprocess_dataset_matching_training(
        test_dataset,
        tokenizer,
        script_args,
        use_survey_contexts=args.use_survey_contexts,
        demo_datasets=demo_datasets,
        num_contexts=args.num_contexts,
        random_seed=args.context_seed,
    )
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
