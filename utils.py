#!/usr/bin/env python3
"""
Utility functions for consistent dataset handling across evaluation scripts.

This module provides standardized functions for:
- Dataset loading from survey and training data
- Common classes and data structures
- Model checkpoint loading
- Dataset preprocessing
"""

import json
import random
import numpy as np
import os
import torch
from typing import List, Dict, Tuple, Union, Any
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from pathlib import Path
from copy import deepcopy
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader




def load_demo_from_survey_and_validation_from_train(
    survey_data_path: str,
    train_data_path: str,
    subsets: List[str],
    demo_size: int = 50,
    validation_size: int = 50,
    seed: int = 0,
    load_dem_only: bool = False,
) -> Tuple[Dict[str, Dataset], Dict[str, Dataset]]:
    """Load demo set from survey file and validation set from train.jsonl.

    This function:
    - Loads demo set from survey file (has embeddings for context sampling)
    - Loads validation set from train.jsonl (has chosen/rejected responses for evaluation)
    - Demo set is sampled using the SAME method as load_and_split_survey_data()

    Args:
        survey_data_path: Base path to survey data (e.g., /path/to/UltraFeedback_single_P_4)
        train_data_path: Base path to train data (e.g., /path/to/P_4_survey_100/gpt2)
        subsets: List of subsets to load (e.g., ['8', '4', '2', '1'])
        demo_size: Number of samples for demo set from survey (default: 50)
        validation_size: Number of samples for validation set from train (default: 50)
        seed: Random seed for reproducibility (default: 0)
        load_dem_only: If True, skip loading validation set from train split (default: False)

    Returns:
        Tuple of (demo_datasets_dict, validation_datasets_dict) keyed by subset
    """
    demo_datasets = {}
    validation_datasets = {}

    for subset in subsets:
        # Load demo from survey file - use same logic as load_and_split_survey_data()
        survey_file = Path(survey_data_path) / f"subset_{subset}_survey_100.jsonl"
        # if not survey_file.exists():
        #     survey_file = Path(survey_data_path) / f"subset_{subset}_survey_47.jsonl"

        if not survey_file.exists():
            raise ValueError(f"Survey file not found: {survey_file}")

        print(f"Loading demo set from survey: {survey_file}")
        survey_dataset = load_dataset('json', data_files=str(survey_file), split=None)

        # Handle DatasetDict
        if isinstance(survey_dataset, dict):
            survey_dataset = survey_dataset['train'] if 'train' in survey_dataset else list(survey_dataset.values())[0]

        # Add data_subset field
        survey_dataset = survey_dataset.map(lambda x: {**x, "data_subset": subset})
        print(f"  Loaded {len(survey_dataset)} samples from survey")

        # Sample demo set - shuffle once and take first demo_size samples
        # This matches the deterministic sampling in load_and_split_survey_data()
        survey_dataset = survey_dataset.shuffle(seed=seed)
        demo_dataset = survey_dataset.select(range(min(demo_size, len(survey_dataset))))
        print(f"  Demo set: {len(demo_dataset)} samples")

        # Load validation from train.jsonl (unless load_dem_only is True)
        if not load_dem_only:
            train_dataset = load_split_dataset(train_data_path, split="train", subset=subset)
            print(f"  Loaded {len(train_dataset)} samples from train")

            # Split train to get validation set (take random validation_size samples)
            train_dataset = train_dataset.shuffle(seed=seed)
            val_dataset = train_dataset.select(range(min(validation_size, len(train_dataset))))
            print(f"  Validation set: {len(val_dataset)} samples")
            validation_datasets[subset] = val_dataset
        else:
            print(f"  Skipping validation set loading (load_dem_only=True)")

        demo_datasets[subset] = demo_dataset

    return demo_datasets, validation_datasets


# ===========================
# Common Classes and Functions
# ===========================


@dataclass
class ScriptArguments:
    """Arguments for inference and optimization scripts."""
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
    """Data collator for reward model inference/evaluation with padding."""

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


def load_split_dataset(data_path: str, split: str = "test", subset: str = "8") -> Dataset:
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
    """Load the VAE model from checkpoint.

    Requires the VAEModel class to be imported from vae_utils.
    """
    # Import VAEModel locally to avoid circular imports
    import sys
    sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')
    from vae_utils import VAEModel

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


def apply_context_to_dataset(dataset_data: Union[List[Dict], Dataset], context: List[Dict]) -> Union[List[Dict], Dataset]:
    """Apply fixed context to all samples in a dataset."""
    if isinstance(dataset_data, Dataset):
        # For HuggingFace Dataset
        def apply_context_fn(example):
            contexts_embeddings = [
                {
                    "embedding_chosen": ctx["embeddings"]["embedding_chosen"],
                    "embedding_rejected": ctx["embeddings"]["embedding_rejected"],
                }
                for ctx in context
            ]
            example["contexts"] = contexts_embeddings
            return example

        return dataset_data.map(apply_context_fn)
    else:
        # For list of dicts
        updated_data = []
        for item in dataset_data:
            updated_item = deepcopy(item)
            contexts_embeddings = [
                {
                    "original_idx": ctx.get("original_idx"),
                    "embedding_chosen": ctx["embeddings"]["embedding_chosen"],
                    "embedding_rejected": ctx["embeddings"]["embedding_rejected"],
                }
                for ctx in context
            ]
            updated_item["contexts"] = contexts_embeddings
            updated_item["context_length"] = len(contexts_embeddings)
            updated_data.append(updated_item)
        return updated_data


def create_demo_based_context(demo_data: List[Dict]) -> List[Dict]:
    """Create context examples from demo data with contradictory flipped versions.

    Takes 50 demo examples and returns 100 total:
    - Original 50 examples as-is (chosen is preferred)
    - 50 flipped examples where chosen and rejected are swapped (rejected is preferred, contradicting the original)

    Args:
        demo_data: List of demo examples, each with 'chosen', 'rejected', and 'embeddings' fields

    Returns:
        List of 100 context examples (50 original + 50 flipped)
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


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    """Move batch tensors to specified device.

    Args:
        batch: Dictionary containing batch data with tensors and other types
        device: Device to move tensors to (e.g., "cuda", "cpu")

    Returns:
        Dictionary with all tensors moved to the specified device
    """
    batch_on_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_on_device[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            batch_on_device[k] = [t.to(device) if isinstance(t, torch.Tensor) else t for t in v]
        else:
            batch_on_device[k] = v
    return batch_on_device


class EvalPreprocessor:
    """Preprocessor for evaluation datasets with tokenization and context extraction."""

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


def preprocess_validation_dataset(
    validation_data: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
) -> Dataset:
    """Preprocess validation dataset matching training format."""

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


def _compute_baseline_logprobs(
    model,
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
    )

    # Convert the dataset properly
    baseline_dict = {
        "chosen": [item["chosen"] for item in baseline_val_data],
        "rejected": [item["rejected"] for item in baseline_val_data],
        "contexts": [item["contexts"] for item in baseline_val_data],
        "data_subset": [item["data_subset"] for item in baseline_val_data],
    }
    baseline_dataset = Dataset.from_dict(baseline_dict)

    original_columns = baseline_dataset.column_names
    baseline_dataset = baseline_dataset.map(
        EvalPreprocessor(args, tokenizer, truncation=True, max_length=args.max_length),
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
            batch_on_device = move_batch_to_device(batch, device)

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


def evaluate_context(
    model,
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
            batch_on_device = move_batch_to_device(batch, device)

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


def save_jsonl(data: List[Dict], output_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


