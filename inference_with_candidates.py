#!/usr/bin/env python3
"""
Inference script for VAE preference model using candidate contexts.

This script:
1. Generates candidate context sets from demo data
2. Evaluates each candidate on the full test set
3. Computes accuracy for each candidate
4. Averages the accuracies across all candidates
5. Saves detailed results and metrics
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
import random
import pickle

# Add path for imports
sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

from vae_utils import VAEModel
from dataset_utils import load_demo_from_survey_and_validation_from_train


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
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})


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
                            "embedding_chosen": context["embeddings"]["embedding_chosen"],
                            "embedding_rejected": context["embeddings"]["embedding_rejected"]
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
                            "embedding_chosen": context["embeddings"]["embedding_chosen"],
                            "embedding_rejected": context["embeddings"]["embedding_rejected"]
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
    # Convert VAE model to bfloat16 to match encoder dtypes
    vae_model = vae_model.to(torch.bfloat16)
    return vae_model, tokenizer


def create_candidate_contexts(demo_data: List[Dict], num_candidates: int = 5, context_length: int = 8, seed: int = 42) -> List[List[Dict]]:
    """Bootstrap multiple candidate context sets from demo data."""
    candidates = []
    random.seed(seed)

    for i in range(num_candidates):
        # Randomly sample context_length items from demo_data
        print(f"Bootstrapping context {i}")
        context = random.sample(demo_data, min(context_length, len(demo_data)))
        candidates.append(context)

    return candidates




def apply_context_to_dataset(dataset: Dataset, context: List[Dict]) -> Dataset:
    """Apply fixed context to all samples in a dataset."""
    def apply_context_fn(example):
        # Create context_embeddings from the fixed context
        contexts_embeddings = [
            {
                "embedding_chosen": ctx["embeddings"]["embedding_chosen"],
                "embedding_rejected": ctx["embeddings"]["embedding_rejected"],
            }
            for ctx in context
        ]
        example["contexts"] = contexts_embeddings
        return example

    return dataset.map(apply_context_fn)


def run_inference(
    model: VAEModel,
    test_dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run inference on test data with candidate context."""
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
                    print(f"    embeddings_chosen shape: {embeddings_chosen.shape}, dtype: {embeddings_chosen.dtype}, mean: {embeddings_chosen.mean():.4f}, std: {embeddings_chosen.std():.4f}")
                    print(f"    embeddings_rejected shape: {embeddings_rejected.shape}, dtype: {embeddings_rejected.dtype}, mean: {embeddings_rejected.mean():.4f}, std: {embeddings_rejected.std():.4f}")
                    print(f"    contexts_chosen shape: {contexts_embeddings_chosen.shape}, dtype: {contexts_embeddings_chosen.dtype}")
                    print(f"    contexts_rejected shape: {contexts_embeddings_rejected.shape}, dtype: {contexts_embeddings_rejected.dtype}")
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
    parser = argparse.ArgumentParser(description="Generate candidates and evaluate on test data")
    parser.add_argument(
        "--subset",
        type=str,
        default="8",
        choices=["1", "2", "4", "8"],
        help="Data subset to evaluate: '1', '2', '4', or '8'. Default: 8"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Number of candidate contexts to generate and evaluate. Default: 50"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=8,
        help="Number of context examples per candidate. Default: 8"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42"
    )
    args = parser.parse_args()

    # Configuration
    data_subset = args.subset
    seed = args.seed
    test_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    train_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    survey_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/UltraFeedback_single_P_4"
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"
    output_dir = f"/hpc/group/fanglab/xx102/vpl_llm/candidate_evaluation_subset_{data_subset}_seed{seed}"

    # Fixed parameters
    num_candidates = args.num_candidates
    context_length = args.context_length
    demo_size = 50

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Data subset: {data_subset}")
    print(f"Random seed: {seed}")
    print(f"Number of candidates: {num_candidates}")
    print(f"Context length: {context_length}")

    # Setup arguments - matching training script arguments
    script_args = ScriptArguments()
    script_args.max_length = 1024
    script_args.per_device_eval_batch_size = 1
    script_args.fixed_contexts = True
    script_args.fixed_llm_embeddings = False
    script_args.other_subsets = "single"
    script_args.controversial_only = True
    script_args.seed = seed

    # Load demo data
    print("\n=== Loading Demo Set ===")
    demo_datasets, _ = load_demo_from_survey_and_validation_from_train(
        survey_data_path,
        train_data_path,
        subsets=[data_subset],
        demo_size=demo_size,
        validation_size=1,
        seed=seed,
    )
    demo_data = list(demo_datasets[data_subset]) if hasattr(demo_datasets[data_subset], '__iter__') else demo_datasets[data_subset]
    print(f"Demo samples: {len(demo_data)}")

    # Load test data
    print("\n=== Loading Test Data ===")
    test_dataset = load_custom_dataset(test_data_path, split="test", subset=data_subset)
    print(f"Loaded {len(test_dataset)} test samples")

    # Filter for controversial only
    if script_args.controversial_only:
        print("\n=== Filtering to Controversial Examples ===")
        test_dataset = test_dataset.filter(lambda example: example.get('controversial', False) == True)
        print(f"After filtering to controversial: {len(test_dataset)} samples")

    # Generate candidate contexts
    print("\n=== Generating Candidate Contexts ===")
    candidates = create_candidate_contexts(demo_data, num_candidates=num_candidates, context_length=context_length, seed=seed)
    print(f"Generated {len(candidates)} candidate context sets")

    # Load checkpoint
    print("\n=== Loading VAE Model ===")
    vae_model, tokenizer = load_checkpoint(
        checkpoint_path,
        model_name="gpt2",
        encoder_embed_dim=768,
        decoder_embed_dim=768,
        hidden_dim=512,
        latent_dim=512,
    )


    # Evaluate each candidate on test data
    print("\n=== Evaluating Candidates on Test Data ===")
    results = []
    accuracies = []

    for candidate_idx, candidate_context in enumerate(candidates):
        print(f"\nEvaluating candidate {candidate_idx + 1}/{len(candidates)}...")

        # Apply this candidate context to test data
        test_with_context = apply_context_to_dataset(test_dataset, candidate_context)

        # Preprocess for evaluation
        test_preprocessed = preprocess_dataset_matching_training(test_with_context, tokenizer, script_args, best_context=candidate_context)
        print(f"  Preprocessed test set: {len(test_preprocessed)} samples")

        # Create data loader
        data_collator = RewardDataCollatorWithPadding(
            args=script_args,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            pad_to_multiple_of=64,
        )

        test_dataloader = DataLoader(
            test_preprocessed,
            batch_size=script_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        # Run inference
        metrics, _ = run_inference(vae_model, test_dataloader, device=device)
        accuracy = metrics["accuracy"]
        print(f"  Test Accuracy: {accuracy:.6f}")

        accuracies.append(accuracy)
        results.append({
            "candidate_idx": candidate_idx,
            "test_accuracy": accuracy,
            "context_indices": [ctx["original_idx"] for ctx in candidate_context],
        })

    # Compute statistics
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    min_accuracy = np.min(accuracies)
    max_accuracy = np.max(accuracies)

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Average Test Accuracy: {avg_accuracy:.6f} ± {std_accuracy:.6f}")
    print(f"Min Accuracy: {min_accuracy:.6f}")
    print(f"Max Accuracy: {max_accuracy:.6f}")

    print("\n=== Individual Results (sorted by Accuracy) ===")
    for result in sorted(results, key=lambda x: x["test_accuracy"], reverse=True):
        print(f"Candidate {result['candidate_idx']:2d}: Accuracy={result['test_accuracy']:8.4f}")

    # Save results
    results_path = os.path.join(output_dir, "candidate_evaluation_results.json")
    with open(results_path, 'w') as f:
        results_to_save = {
            "num_candidates": num_candidates,
            "context_length": context_length,
            "data_subset": data_subset,
            "average_test_accuracy": float(avg_accuracy),
            "std_test_accuracy": float(std_accuracy),
            "min_test_accuracy": float(min_accuracy),
            "max_test_accuracy": float(max_accuracy),
            "individual_results": results,
            "all_accuracies": [float(x) for x in accuracies],
        }
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save candidates for later use
    candidates_path = os.path.join(output_dir, "candidates.pkl")
    with open(candidates_path, 'wb') as f:
        pickle.dump(candidates, f)
    print(f"Candidates saved to {candidates_path}")

    print("\n✓ Candidate evaluation complete!")
    print(f"Average Test Accuracy: {avg_accuracy:.6f}")
    print(f"Standard Deviation: {std_accuracy:.6f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
