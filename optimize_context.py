#!/usr/bin/env python3
"""
Script to optimize context selection for VAE preference model.

This script:
1. Splits survey_100.jsonl into 10 demo and 50 validation samples
2. Bootstraps multiple candidate context sets from demo
3. Evaluates each candidate on validation set using VAE model
4. Selects the best performing context
5. Applies it to all test examples
"""

import os
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader
from copy import deepcopy
import sys
import random

# Add path for imports
sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

from vae_utils import VAEModel
from dataset_utils import split_dataset_simple, load_demo_from_survey_and_validation_from_train


@dataclass
class ScriptArguments:
    """Arguments for the optimization script."""
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
    """Data collator for reward model evaluation with padding."""

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


def load_survey_data(survey_path: str) -> List[Dict]:
    """Load survey JSONL file."""
    data = []
    with open(survey_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


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


def preprocess_validation_dataset(
    validation_data: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
) -> Dataset:
    """Preprocess validation dataset matching training format."""

    class EvalPreprocessor:
        def __init__(self, args, tokenizer, **tokenizer_kwargs):
            self.tokenizer = tokenizer
            self.args = args
            self.tokenizer_kwargs = tokenizer_kwargs

        def __call__(self, examples):
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

    # Convert to HF Dataset
    dataset_dict = {
        "chosen": [item["chosen"] for item in validation_data],
        "rejected": [item["rejected"] for item in validation_data],
        "contexts": [item["contexts"] for item in validation_data],
        "data_subset": [item["data_subset"] for item in validation_data],
    }
    dataset = Dataset.from_dict(dataset_dict)

    original_columns = dataset.column_names
    dataset = dataset.map(
        EvalPreprocessor(args, tokenizer, truncation=True, max_length=args.max_length),
        batched=True,
        num_proc=10,
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


def create_demo_based_context(demo_data: List[Dict]) -> List[Dict]:
    """Create context examples from demo data with contradictory flipped versions.

    Takes 20 demo examples and returns 40 total:
    - Original 20 examples as-is (chosen is preferred)
    - 20 flipped examples where chosen and rejected are swapped (rejected is preferred, contradicting the original)

    Args:
        demo_data: List of demo examples, each with 'chosen', 'rejected', and 'embeddings' fields

    Returns:
        List of 40 context examples (20 original + 20 flipped)
    """
    contexts = []

    # Add original examples
    for i, example in enumerate(demo_data):
        context = {
            "original_idx": f"demo_{i}",
            "embeddings": {
                "embedding_chosen": example["embeddings"]["embedding_chosen"],
                "embedding_rejected": example["embeddings"]["embedding_rejected"],
            }
        }
        contexts.append(context)

    # Add flipped examples (contradictory preferences)
    for i, example in enumerate(demo_data):
        context = {
            "original_idx": f"demo_flipped_{i}",
            "embeddings": {
                "embedding_chosen": example["embeddings"]["embedding_rejected"],
                "embedding_rejected": example["embeddings"]["embedding_chosen"],
            }
        }
        contexts.append(context)

    return contexts


def evaluate_context(
    model: VAEModel,
    validation_dataset: Dataset,
    validation_data: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    device: str = "cuda",
    baseline_contexts: List[Dict] = None,
) -> tuple:
    """Evaluate a context set on validation dataset using gain-based SNR metric.

    Gain is computed as:
    gain_i = log P(yi=yi_hat|x_i, context) - log P(yi=yi_hat|x_i)

    SNR is computed as:
    SNR = mean(gain_i) / std(gain_i)

    where log P(true|x) = log_sigmoid(reward_score)

    The baseline log P(yi=yi_hat|x_i) is approximated using random meaningless contexts.
    """

    data_collator = RewardDataCollatorWithPadding(
        args=args,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=64,
    )

    model = model.to(device)
    model.eval()

    # First, evaluate on baseline (random contexts) to get baseline log probs
    print("  Computing baseline (no context)...")
    baseline_log_probs = _compute_baseline_logprobs(
        model, validation_data, tokenizer, args, baseline_contexts, device
    )

    # Then, evaluate with actual context
    print("  Computing with context...")
    dataloader = DataLoader(
        validation_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    accuracies = []
    log_probs_chosen = []
    log_probs_rejected = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
                user_type_batch = torch.tensor(batch_on_device["user_type"], dtype=torch.bfloat16).to(device)

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

                rewards_chosen, rewards_rejected, _, _, _ = model(
                    embeddings_chosen,
                    embeddings_rejected,
                    contexts_embeddings_chosen,
                    contexts_embeddings_rejected,
                    seq_start_end,
                    user_type_batch,
                    False,
                )

                # Compute normalized log probabilities using log_softmax
                rewards = torch.stack([rewards_chosen, rewards_rejected], dim=-1)
                log_probs = torch.nn.functional.log_softmax(rewards, dim=-1)
                log_prob_chosen = log_probs[..., 0].float().cpu().numpy().flatten()
                log_prob_rejected = log_probs[..., 1].float().cpu().numpy().flatten()

                log_probs_chosen.extend(log_prob_chosen)
                log_probs_rejected.extend(log_prob_rejected)

                accuracy = (rewards_chosen > rewards_rejected).float().mean().item()
                accuracies.append(accuracy)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    avg_accuracy = np.mean(accuracies) if accuracies else 0.0

    # Compute gains: difference between context-enhanced and baseline predictions
    log_probs_chosen = np.array(log_probs_chosen)
    baseline_chosen = np.array(baseline_log_probs["chosen"])

    gains = log_probs_chosen - baseline_chosen

    # Compute SNR on gains
    mean_gain = np.mean(gains)
    std_gain = np.std(gains)

    # SNR = signal_strength / signal_variance
    snr = mean_gain / (std_gain + 1e-8)  # Add small epsilon to avoid division by zero

    return float(avg_accuracy), float(snr)


def _compute_baseline_logprobs(
    model: VAEModel,
    validation_data: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    baseline_contexts: List[Dict],
    device: str = "cuda",
) -> Dict[str, List]:
    """Compute baseline log probabilities using provided baseline contexts."""

    # Create a validation dataset with baseline contexts
    baseline_val_data = apply_context_to_dataset(
        validation_data,
        baseline_contexts,
        context_length=len(baseline_contexts)
    )

    # Convert the dataset properly
    baseline_dict = {
        "chosen": [item["chosen"] for item in baseline_val_data],
        "rejected": [item["rejected"] for item in baseline_val_data],
        "contexts": [item["contexts"] for item in baseline_val_data],
        "data_subset": [item["data_subset"] for item in baseline_val_data],
    }
    baseline_dataset = Dataset.from_dict(baseline_dict)

    class BaselinePreprocessor:
        def __init__(self, args, tokenizer, baseline_contexts, **tokenizer_kwargs):
            self.tokenizer = tokenizer
            self.args = args
            self.tokenizer_kwargs = tokenizer_kwargs
            self.baseline_contexts = baseline_contexts

        def __call__(self, examples):
            new_examples = {
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": [],
                "contexts_embeddings": [],
            }
            for chosen, rejected, contexts, user_type in zip(
                examples["chosen"], examples["rejected"], examples["contexts"], examples["data_subset"]
            ):
                tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
                tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

                new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
                new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
                new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
                new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

                contexts_embeddings = [
                    {
                        "embedding_chosen": context["embedding_chosen"],
                        "embedding_rejected": context["embedding_rejected"]
                    }
                    for context in contexts
                ]
                new_examples["contexts_embeddings"].append(contexts_embeddings)

            new_examples["user_type"] = examples["data_subset"]
            return new_examples

    original_columns = baseline_dataset.column_names
    baseline_dataset = baseline_dataset.map(
        BaselinePreprocessor(args, tokenizer, baseline_contexts, truncation=True, max_length=args.max_length),
        batched=True,
        num_proc=10,
        remove_columns=original_columns,
    )

    data_collator = RewardDataCollatorWithPadding(
        args=args,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=64,
    )

    dataloader = DataLoader(
        baseline_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    baseline_log_probs = {"chosen": [], "rejected": []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
                user_type_batch = torch.tensor(batch_on_device["user_type"], dtype=torch.bfloat16).to(device)

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

                rewards_chosen, rewards_rejected, _, _, _ = model(
                    embeddings_chosen,
                    embeddings_rejected,
                    contexts_embeddings_chosen,
                    contexts_embeddings_rejected,
                    seq_start_end,
                    user_type_batch,
                    False,
                )

                # Compute normalized log probabilities using log_softmax
                rewards = torch.stack([rewards_chosen, rewards_rejected], dim=-1)
                log_probs = torch.nn.functional.log_softmax(rewards, dim=-1)
                log_prob_chosen = log_probs[..., 0].float().cpu().numpy().flatten()
                log_prob_rejected = log_probs[..., 1].float().cpu().numpy().flatten()

                baseline_log_probs["chosen"].extend(log_prob_chosen)
                baseline_log_probs["rejected"].extend(log_prob_rejected)

            except Exception as e:
                print(f"Error processing baseline batch {batch_idx}: {e}")
                continue

    return baseline_log_probs


def apply_context_to_dataset(dataset_data: List[Dict], context: List[Dict], context_length: int = 8) -> List[Dict]:
    """Apply fixed context to all samples in a dataset."""
    updated_data = []

    for item in dataset_data:
        updated_item = deepcopy(item)
        # Create context_embeddings from the fixed context
        contexts_embeddings = [
            {
                "original_idx": ctx["original_idx"],
                "embedding_chosen": ctx["embeddings"]["embedding_chosen"],
                "embedding_rejected": ctx["embeddings"]["embedding_rejected"],
            }
            for ctx in context
        ]

        # Update the contexts field
        updated_item["contexts"] = contexts_embeddings
        updated_item["context_length"] = len(contexts_embeddings)
        updated_data.append(updated_item)

    return updated_data


def save_jsonl(data: List[Dict], output_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Optimize context selection for VAE preference model")
    parser.add_argument(
        "--subset",
        type=str,
        default="8",
        choices=["1", "2", "4", "8"],
        help="Data subset to optimize context for: '1', '2', '4', or '8'. Default: 8"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42"
    )
    cmd_args = parser.parse_args()

    # Configuration
    data_subset = cmd_args.subset
    seed = cmd_args.seed
    survey_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/UltraFeedback_single_P_4"
    train_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"
    output_dir = f"/hpc/group/fanglab/xx102/vpl_llm/context_optimization_subset_{data_subset}_seed{seed}"

    # Fixed parameters
    context_length = 8
    num_candidates = 20
    demo_size = 50  # Demo set for bootstrapping contexts (from survey)
    validation_size = 50  # Validation set for evaluation (from train.jsonl)

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Data subset: {data_subset}")
    print(f"Random seed: {seed}")
    print(f"Context length: {context_length}")

    # Setup arguments
    args = ScriptArguments()
    args.max_length = 1024
    args.per_device_eval_batch_size = 1
    args.fixed_contexts = True
    args.fixed_llm_embeddings = False
    args.other_subsets = "single"
    args.seed = seed

    # Load demo from survey and validation from train
    print("\n=== Loading Demo Set (from Survey) and Validation Set (from Train) ===")
    demo_datasets, validation_datasets = load_demo_from_survey_and_validation_from_train(
        survey_data_path,
        train_data_path,
        subsets=[data_subset],
        demo_size=demo_size,
        validation_size=validation_size,
        seed=seed,
    )

    demo_data = list(demo_datasets[data_subset]) if hasattr(demo_datasets[data_subset], '__iter__') else demo_datasets[data_subset]
    validation_data = list(validation_datasets[data_subset]) if hasattr(validation_datasets[data_subset], '__iter__') else validation_datasets[data_subset]
    print(f"Demo samples: {len(demo_data)}")
    print(f"Validation samples: {len(validation_data)}")

    # Create candidate contexts
    print("\n=== Creating Candidate Contexts ===")
    candidates = create_candidate_contexts(demo_data, num_candidates=num_candidates, context_length=context_length, seed=seed)
    print(f"Created {len(candidates)} candidate context sets")

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

    # Create baseline contexts from demo data (20 original + 20 flipped for contradictory preferences)
    print("\n=== Creating Baseline Contexts from Demo Data ===")
    baseline_contexts = create_demo_based_context(demo_data)
    print(f"Created {len(baseline_contexts)} baseline contexts from demo data ({len(demo_data)} original + {len(demo_data)} flipped)")

    # Evaluate each candidate
    print("\n=== Evaluating Candidates ===")
    results = []

    for candidate_idx, candidate_context in enumerate(candidates):
        print(f"\nEvaluating candidate {candidate_idx + 1}/{len(candidates)}...")

        # Apply this candidate context to validation data
        val_with_context = apply_context_to_dataset(validation_data, candidate_context)

        # Preprocess for evaluation
        val_dataset = preprocess_validation_dataset(val_with_context, tokenizer, args)
        print(f"  Preprocessed validation set: {len(val_dataset)} samples")

        # Evaluate with gain-based metric
        accuracy, snr = evaluate_context(vae_model, val_dataset, val_with_context, tokenizer, args, device, baseline_contexts)
        print(f"  Accuracy: {accuracy:.6f}")
        print(f"  SNR (gain-based): {snr:.6f}")

        results.append({
            "candidate_idx": candidate_idx,
            "accuracy": accuracy,
            "snr": snr,
            "context_indices": [ctx["original_idx"] for ctx in candidate_context],
        })

    # Find best candidate by SNR
    print("\n=== Results ===")
    best_result = max(results, key=lambda x: x["snr"])
    print(f"Best candidate: {best_result['candidate_idx']}")
    print(f"  Accuracy: {best_result['accuracy']:.6f}")
    print(f"  SNR: {best_result['snr']:.6f}")
    print(f"  Context indices: {best_result['context_indices']}")

    # Print all results for comparison
    print("\n=== All Results (sorted by SNR) ===")
    for result in sorted(results, key=lambda x: x["snr"], reverse=True):
        print(f"Candidate {result['candidate_idx']:2d}: SNR={result['snr']:8.4f}, Accuracy={result['accuracy']:8.4f}")

    # Save results
    results_path = os.path.join(output_dir, "optimization_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save best context
    best_context = candidates[best_result["candidate_idx"]]
    best_context_path = os.path.join(output_dir, "best_context.json")
    with open(best_context_path, 'w') as f:
        json.dump([{k: v for k, v in item.items() if k != "embeddings"} for item in best_context], f, indent=2)
    print(f"Best context saved to {best_context_path}")

    # Save best context with embeddings for later application
    best_context_full_path = os.path.join(output_dir, "best_context_with_embeddings.pkl")
    import pickle
    with open(best_context_full_path, 'wb') as f:
        pickle.dump(best_context, f)
    print(f"Best context with embeddings saved to {best_context_full_path}")

    print("\n✓ Context optimization complete!")
    print(f"Best SNR: {best_result['snr']:.6f}")
    print(f"Best accuracy: {best_result['accuracy']:.6f}")
    print(f"Best candidate index: {best_result['candidate_idx']}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo apply this context to test data, run:")
    print(f"  python3 apply_optimized_context.py {best_result['candidate_idx']}")


if __name__ == "__main__":
    main()
