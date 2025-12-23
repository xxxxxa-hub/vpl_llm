#!/usr/bin/env python3
"""
Inference script for VAE preference model using optimized context.

This script:
1. Loads the best optimized context from context_optimization/best_context.json
2. Applies it to test data (if needed)
3. Runs inference on test set using the trained VAE model
4. Saves predictions and metrics
"""

import os
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader
from copy import deepcopy
import sys

# Add path for imports
sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

from vae_utils import VAEModel


@dataclass
class ScriptArguments:
    """Arguments for the inference script."""
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_eval_batch_size: int = field(default=1)
    max_length: int = field(default=1024)
    fixed_contexts: bool = field(default=True)
    fixed_llm_embeddings: bool = field(default=False)
    other_subsets: str = field(default="single")
    tokenizer_name: str = field(default=None)
    controversial_only: bool = field(default=False)


class RewardDataCollatorWithPadding:
    """Data collator for reward model inference with padding."""

    def __init__(
        self,
        args: ScriptArguments,
        tokenizer: PreTrainedTokenizerBase,
        padding: str = True,
        max_length: int = None,
        pad_to_multiple_of: int = None,
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
        user_mapping = {
            "8": 0,
            "4": 1,
            "2": 2,
            "1": 3,
        }

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
        elif self.args.fixed_contexts:
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


def load_custom_dataset(data_path: str, split: str = "test", subset: str = "8") -> Dataset:
    """Load dataset from custom JSONL files for a specific subset."""
    base_path = Path(data_path)
    subset_dir = base_path / subset
    split_file = subset_dir / f"{split}.jsonl"

    if not split_file.exists():
        raise ValueError(f"Dataset file not found: {split_file}")

    print(f"Loading {split_file}")
    dataset = load_dataset('json', data_files=str(split_file), split=None)

    if isinstance(dataset, dict):
        dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]

    dataset = dataset.map(lambda x: {**x, "data_subset": subset})
    print(f"  Loaded {len(dataset)} examples from {subset}")

    return dataset


def preprocess_dataset_matching_training(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    best_context=None,
    num_proc: int = 24
) -> Dataset:
    """Preprocess dataset to match training configuration.

    Args:
        best_context: If provided, replace all contexts with this optimized context
    """

    class EvalHHRLHFPreprocessor:
        def __init__(self, args, tokenizer, best_context=None, **tokenizer_kwargs):
            self.tokenizer = tokenizer
            self.args = args
            self.tokenizer_kwargs = tokenizer_kwargs
            self.best_context = best_context

        def __call__(self, examples):
            if self.args.fixed_llm_embeddings:
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

                    # Use best context if provided, otherwise use example contexts
                    contexts_to_use = self.best_context if self.best_context else contexts
                    contexts_embeddings = [
                        {
                            "embedding_chosen": context["embeddings"]["embedding_chosen"] if self.best_context else context["embedding_chosen"],
                            "embedding_rejected": context["embeddings"]["embedding_rejected"] if self.best_context else context["embedding_rejected"]
                        }
                        for context in contexts_to_use
                    ]
                    new_examples["contexts_embeddings"].append(contexts_embeddings)
                    new_examples["max_lengths"].append(0)
                new_examples["user_type"] = examples["data_subset"]
                return new_examples
            else:
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

                    # Use best context if provided, otherwise use example contexts
                    contexts_to_use = self.best_context if self.best_context else contexts
                    contexts_embeddings = [
                        {
                            "embedding_chosen": context["embeddings"]["embedding_chosen"] if self.best_context else context["embedding_chosen"],
                            "embedding_rejected": context["embeddings"]["embedding_rejected"] if self.best_context else context["embedding_rejected"]
                        }
                        for context in contexts_to_use
                    ]
                    new_examples["contexts_embeddings"].append(contexts_embeddings)
                    new_examples["max_lengths"].append(max_length)

                new_examples["user_type"] = examples["data_subset"]
                return new_examples

    original_columns = dataset.column_names

    dataset = dataset.map(
        EvalHHRLHFPreprocessor(args, tokenizer, best_context=best_context, truncation=True, max_length=args.max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

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

    llm_encoder = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=decoder_embed_dim, torch_dtype=torch.bfloat16
    )
    llm_encoder.score.weight.data *= 0.01

    contexts_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=encoder_embed_dim, torch_dtype=torch.bfloat16
    )
    contexts_model.score.weight.data *= 0.01

    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")

    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        print("Loading PEFT adapters from checkpoint...")
        try:
            llm_encoder = PeftModel.from_pretrained(llm_encoder, checkpoint_path)
            llm_encoder.eval()
            contexts_model = PeftModel.from_pretrained(contexts_model, checkpoint_path)
            contexts_model.eval()
        except Exception as e:
            print(f"Warning: Error loading PEFT adapters: {e}")
            print("Continuing with base models...")
    else:
        print("No PEFT adapters found. Using base models.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_encoder.config.pad_token_id = tokenizer.pad_token_id
    contexts_model.config.pad_token_id = tokenizer.pad_token_id

    model_pt_path = os.path.join(checkpoint_path, "model.pt")
    if os.path.exists(model_pt_path):
        print(f"Loading VAE model from {model_pt_path}")
        vae_model = torch.load(model_pt_path, map_location='cpu', weights_only=False)
        vae_model.llm_encoder = llm_encoder
        vae_model.llm_contexts_encoder = contexts_model
    else:
        print(f"Creating VAE model with base encoders...")
        vae_model = VAEModel(
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            llm_encoder=llm_encoder,
            llm_contexts_encoder=contexts_model,
            fixed_contexts=True,
            fixed_llm_embeddings=False,
        )

    vae_model.eval()
    return vae_model, tokenizer


def run_inference(
    model: VAEModel,
    test_dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run inference on test data with optimized context."""
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

    print(f"Running inference on {device}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}...")

            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    batch_on_device[k] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
                else:
                    batch_on_device[k] = v

            try:
                seq_start_end = batch_on_device["seq_start_end"].to(device)
                user_type_batch = torch.tensor(batch_on_device["user_type"], dtype=torch.float32).to(device)

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

                # Forward pass through VAE model
                rewards_chosen, rewards_rejected, mean, log_var, z = model(
                    embeddings_chosen,
                    embeddings_rejected,
                    contexts_embeddings_chosen,
                    contexts_embeddings_rejected,
                    seq_start_end,
                    user_type_batch,
                    False,
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

            user_type = batch_on_device["user_type"]

            # Store results
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
    parser = argparse.ArgumentParser(description="Run inference with optimized or refined context")
    parser.add_argument(
        "--context",
        type=str,
        choices=["initial", "refined"],
        default="initial",
        help="Which context to use: 'initial' (optimization) or 'refined' (after refinement). Default: initial"
    )
    args = parser.parse_args()

    # Configuration - using optimized context
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"
    test_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    context_optimization_dir = "/hpc/group/fanglab/xx102/vpl_llm/context_optimization"
    context_refinement_dir = "/hpc/group/fanglab/xx102/vpl_llm/context_refinement"
    data_subset = "8"  # Only working with subset '8'

    # Determine which context to use and output directory
    if args.context == "refined":
        context_source_dir = context_refinement_dir
        context_file = "refined_context_with_embeddings.pkl"
        output_dir = "/hpc/group/fanglab/xx102/vpl_llm/inference_results_refined_context"
        context_label = "REFINED"
    else:
        context_source_dir = context_optimization_dir
        context_file = "best_context_with_embeddings.pkl"
        output_dir = "/hpc/group/fanglab/xx102/vpl_llm/inference_results_optimized_context"
        context_label = "INITIAL (OPTIMIZED)"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Data subset: {data_subset}")
    print(f"Context: {context_label}")

    # Setup arguments - matching training script arguments
    script_args = ScriptArguments()
    script_args.max_length = 1024
    script_args.per_device_eval_batch_size = 1
    script_args.fixed_contexts = True
    script_args.fixed_llm_embeddings = False
    script_args.other_subsets = "single"
    script_args.controversial_only = True

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

    # Load test data for subset '8' only
    print("\n=== Loading Test Data ===")
    test_dataset = load_custom_dataset(test_data_path, split="test", subset=data_subset)
    print(f"Loaded {len(test_dataset)} test samples")

    # Filter for controversial only
    if script_args.controversial_only:
        print("\n=== Filtering to Controversial Examples ===")
        test_dataset = test_dataset.filter(lambda example: example.get('controversial', False) == True)
        print(f"After filtering to controversial: {len(test_dataset)} samples")

    # Load context from selected source
    print(f"\n=== Loading {context_label} Context ===")
    import pickle
    best_context_path = os.path.join(context_source_dir, context_file)
    if not os.path.exists(best_context_path):
        print(f"WARNING: Context file not found at {best_context_path}")
        print("Using contexts from test.jsonl (may be random contexts)")
        best_context = None
    else:
        with open(best_context_path, 'rb') as f:
            best_context = pickle.load(f)
        print(f"Loaded {context_label} context with {len(best_context)} samples")

    # Preprocess dataset matching training configuration
    print("\n=== Preprocessing Dataset ===")
    test_dataset = preprocess_dataset_matching_training(test_dataset, tokenizer, script_args, best_context=best_context)
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

    # Run inference
    print("\n=== Running Inference with Optimized Context ===")
    metrics, predictions = run_inference(vae_model, test_dataloader, device=device)

    # Print results
    print("\n=== Inference Results with Optimized Context ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
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
    report_path = os.path.join(output_dir, "inference_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VAE Preference Model Inference Report (Optimized Context)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Data: {test_data_path}\n")
        f.write(f"Optimized Context: {context_optimization_dir}\n")
        f.write(f"Test Samples: {metrics['num_samples']}\n\n")
        f.write("Metrics:\n")
        f.write("-" * 60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<50} {value:.6f}\n")
            else:
                f.write(f"{key:.<50} {value}\n")
    print(f"Report saved to {report_path}")

    print("\n✓ Inference complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
