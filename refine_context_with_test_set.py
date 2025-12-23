#!/usr/bin/env python3
"""
Iterative context refinement using test set examples.

This script:
1. Loads the best context from optimize_context.py
2. Creates an augmented test set by reversing preferred/dispreferred responses
3. Iteratively searches for better contexts by replacing context samples with test examples
4. Evaluates on validation set and keeps improvements
"""

import os
import json
import pickle
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
from tqdm import tqdm

# Add path for imports
sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

from vae_utils import VAEModel


@dataclass
class ScriptArguments:
    """Arguments for the refinement script."""
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_eval_batch_size: int = field(default=1)
    max_length: int = field(default=1024)
    fixed_contexts: bool = field(default=True)
    fixed_llm_embeddings: bool = field(default=False)
    other_subsets: str = field(default="single")
    tokenizer_name: str = field(default=None)
    controversial_only: bool = field(default=False)


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

        if self.args.fixed_contexts:
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


def load_best_context(context_optimization_dir: str):
    """Load the best context from previous optimization."""
    best_context_path = os.path.join(context_optimization_dir, "best_context_with_embeddings.pkl")

    if not os.path.exists(best_context_path):
        raise ValueError(f"Best context file not found: {best_context_path}")

    with open(best_context_path, 'rb') as f:
        best_context = pickle.load(f)

    # Mark all initial contexts as true preferences (not synthesized)
    for ctx in best_context:
        ctx["is_true_preference"] = True

    return best_context


def create_augmented_test_set(test_data: List[Dict]) -> List[Dict]:
    """Create augmented test set by reversing preferred/dispreferred.

    For each example (chosen, rejected), create a reversed example (rejected, chosen).
    This doubles the test set size.
    """
    augmented_data = []

    for item in test_data:
        # Original - mark as true preference
        original_item = deepcopy(item)
        original_item["is_true_preference"] = True
        augmented_data.append(original_item)

        # Reversed: swap chosen and rejected - mark as synthesized
        reversed_item = deepcopy(item)
        reversed_item["chosen"] = item["rejected"]
        reversed_item["rejected"] = item["chosen"]
        reversed_item["is_true_preference"] = False

        # Swap embeddings if they exist
        if "embeddings" in item:
            reversed_item["embeddings"]["embedding_chosen"] = item["embeddings"]["embedding_rejected"]
            reversed_item["embeddings"]["embedding_rejected"] = item["embeddings"]["embedding_chosen"]

        augmented_data.append(reversed_item)

    return augmented_data


def preprocess_validation_dataset(
    validation_data: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    current_context,
    num_proc: int = 1,
) -> Dataset:
    """Preprocess validation dataset with current context.

    Args:
        num_proc: Number of processes for parallel preprocessing (default: 1)
    """

    class EvalPreprocessor:
        def __init__(self, args, tokenizer, current_context, **tokenizer_kwargs):
            self.tokenizer = tokenizer
            self.args = args
            self.tokenizer_kwargs = tokenizer_kwargs
            self.current_context = current_context

        def __call__(self, examples):
            new_examples = {
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": [],
                "contexts_embeddings": [],
                "max_lengths": []
            }
            for chosen, rejected, user_type in zip(
                examples["chosen"], examples["rejected"], examples["data_subset"]
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

                # Use current context
                contexts_embeddings = [
                    {
                        "embedding_chosen": ctx["embeddings"]["embedding_chosen"],
                        "embedding_rejected": ctx["embeddings"]["embedding_rejected"]
                    }
                    for ctx in self.current_context
                ]
                new_examples["contexts_embeddings"].append(contexts_embeddings)
                new_examples["max_lengths"].append(max_length)

            new_examples["user_type"] = examples["data_subset"]
            return new_examples

    # Convert to HF Dataset
    dataset_dict = {
        "chosen": [item["chosen"] for item in validation_data],
        "rejected": [item["rejected"] for item in validation_data],
        "data_subset": [item["data_subset"] for item in validation_data],
    }
    dataset = Dataset.from_dict(dataset_dict)

    original_columns = dataset.column_names
    dataset = dataset.map(
        EvalPreprocessor(args, tokenizer, current_context, truncation=True, max_length=args.max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    dataset = dataset.filter(lambda x: x["max_lengths"] <= args.max_length)

    return dataset


def evaluate_context(
    model: VAEModel,
    validation_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    device: str = "cuda",
) -> tuple:
    """Evaluate a context set on validation dataset and return accuracy and SNR.

    SNR is computed as:
    SNR = mean(log P(true|chosen)) / std(log P(true|chosen))

    where log P(true|x) = log_sigmoid(reward_score)
    Measures how consistently high the model rates chosen responses.
    """

    data_collator = RewardDataCollatorWithPadding(
        args=args,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=64,
    )

    dataloader = DataLoader(
        validation_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    model = model.to(device)
    model.eval()

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

                rewards_chosen, rewards_rejected, _, _, _ = model(
                    embeddings_chosen,
                    embeddings_rejected,
                    contexts_embeddings_chosen,
                    contexts_embeddings_rejected,
                    seq_start_end,
                    user_type_batch,
                    False,
                )

                # Compute log P(true|x) = log_sigmoid(reward) for both chosen and rejected
                log_prob_chosen = torch.nn.functional.logsigmoid(rewards_chosen).float().cpu().numpy().flatten()
                log_prob_rejected = torch.nn.functional.logsigmoid(rewards_rejected).float().cpu().numpy().flatten()

                log_probs_chosen.extend(log_prob_chosen)
                log_probs_rejected.extend(log_prob_rejected)

                accuracy = (rewards_chosen > rewards_rejected).float().mean().item()
                accuracies.append(accuracy)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    avg_accuracy = np.mean(accuracies) if accuracies else 0.0

    # Compute SNR
    log_probs_chosen = np.array(log_probs_chosen)
    log_probs_rejected = np.array(log_probs_rejected)

    mean_chosen = np.mean(log_probs_chosen)
    std_chosen = np.std(log_probs_chosen)

    # SNR = signal_strength / signal_variance
    # Measures how consistently high the model rates chosen responses
    snr = mean_chosen / (std_chosen + 1e-8)  # Add small epsilon to avoid division by zero

    return float(avg_accuracy), float(snr)


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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Refine context using test set examples")
    parser.add_argument(
        "--subset",
        type=str,
        default="8",
        choices=["1", "2", "4", "8"],
        help="Data subset to refine context for: '1', '2', '4', or '8'. Default: 8"
    )
    cmd_args = parser.parse_args()

    # Configuration
    data_subset = cmd_args.subset
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"
    context_optimization_dir = f"/hpc/group/fanglab/xx102/vpl_llm/context_optimization_subset_{data_subset}"
    test_data_path = f"/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2/{data_subset}/test.jsonl"
    survey_path = f"/hpc/group/fanglab/xx102/vpl_llm/data/UltraFeedback_single_P_4/{data_subset}/survey_100.jsonl"
    output_dir = f"/hpc/group/fanglab/xx102/vpl_llm/context_refinement_subset_{data_subset}"

    # Search parameters
    num_iterations = 8
    context_length = 8
    num_proc = 24  # Number of processes for parallel preprocessing (increase for faster preprocessing)

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Data subset: {data_subset}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Context length: {context_length}")
    print(f"Parallel preprocessing: {num_proc} processes")

    # Setup arguments
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

    # Load best context as initialization
    print("\n=== Loading Best Context (Initialization) ===")
    current_context = load_best_context(context_optimization_dir)
    print(f"Loaded best context with {len(current_context)} samples")
    print(f"Context indices: {[ctx['Index'] for ctx in current_context]}")

    # Load test data and create augmented set
    print("\n=== Loading and Augmenting Test Data ===")
    test_data = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test samples")

    augmented_test_data = create_augmented_test_set(test_data)
    print(f"Created augmented test set with {len(augmented_test_data)} samples")

    # Load and prepare validation data
    print("\n=== Loading Validation Data ===")
    with open(survey_path, 'r') as f:
        survey_data = []
        for line in f:
            survey_data.append(json.loads(line))

    # Split survey data into demo and validation (same as optimize_context.py)
    # demo_size=10, validation_size=50 (indices 10:60)
    demo_size = 20
    validation_size = 20
    random.seed(0)
    shuffled_data = survey_data.copy()
    random.shuffle(shuffled_data)
    validation_data = shuffled_data[demo_size:demo_size + validation_size]

    # Add data_subset field
    for item in validation_data:
        item["data_subset"] = data_subset

    print(f"Validation set size: {len(validation_data)}")

    # Evaluate initial context
    print("\n=== Evaluating Initial Context ===")
    val_dataset = preprocess_validation_dataset(validation_data, tokenizer, script_args, current_context, num_proc=num_proc)
    initial_accuracy, initial_snr = evaluate_context(vae_model, val_dataset, tokenizer, script_args, device)
    print(f"Initial context - Accuracy: {initial_accuracy:.6f}, SNR: {initial_snr:.6f}")

    # Iterative search
    print("\n=== Starting Iterative Search ===")
    search_history = []
    best_context = deepcopy(current_context)
    best_accuracy = initial_accuracy
    best_snr = initial_snr

    # Systematic hill-climbing: cycle through positions with local search
    # For each position, sample multiple candidates and pick the best one
    pbar = tqdm(range(num_iterations), desc="Searching for better context")
    for iteration in pbar:
        # Cycle through positions systematically (0-7, then repeat)
        replace_pos = iteration % context_length

        # For this position, try multiple random test samples and pick the best
        num_candidates_per_pos = 10  # Try 10 candidates per position
        sampled_indices = random.sample(range(len(augmented_test_data)), min(num_candidates_per_pos, len(augmented_test_data)))

        best_candidate_for_pos = None
        best_snr_for_pos = best_snr
        best_acc_for_pos = best_accuracy
        best_sample_idx = -1

        print(f"Iteration {iteration}: Position {replace_pos} (best SNR so far: {best_snr:.6f})")

        # Evaluate all candidates for this position
        for candidate_idx, sample_idx in enumerate(sampled_indices, 1):
            sample = augmented_test_data[sample_idx]

            # Create candidate context with replacement at this position
            candidate_context = deepcopy(current_context)
            candidate_context[replace_pos] = sample

            # Evaluate new context
            val_dataset = preprocess_validation_dataset(validation_data, tokenizer, script_args, candidate_context, num_proc=num_proc)
            new_accuracy, new_snr = evaluate_context(vae_model, val_dataset, tokenizer, script_args, device)

            # Print SNR and accuracy for each evaluation
            is_best_for_pos = "★" if new_snr > best_snr_for_pos else " "
            print(f"  [{candidate_idx}/{len(sampled_indices)}] test_idx={sample_idx}: SNR={new_snr:.6f} Acc={new_accuracy:.6f} {is_best_for_pos}")

            # Track best candidate for this position (using SNR)
            if new_snr > best_snr_for_pos:
                best_snr_for_pos = new_snr
                best_acc_for_pos = new_accuracy
                best_candidate_for_pos = candidate_context
                best_sample_idx = sample_idx

        # Log best result for this position
        best_sample_is_true_pref = None
        best_sample_index = None
        best_sample_original_id = None
        if best_sample_idx >= 0 and best_sample_idx < len(augmented_test_data):
            best_sample = augmented_test_data[best_sample_idx]
            best_sample_is_true_pref = best_sample.get("is_true_preference", None)
            best_sample_index = best_sample.get("Index", None)
            best_sample_original_id = best_sample.get("original_id", None)

        result = {
            "iteration": iteration,
            "selected_sample": {
                "augmented_test_data_idx": best_sample_idx,
                "Index": best_sample_index,
                "original_id": best_sample_original_id,
                "is_true_preference": best_sample_is_true_pref
            },
            "replace_pos": replace_pos,
            "old_snr": best_snr,
            "new_snr": best_snr_for_pos,
            "old_accuracy": best_accuracy,
            "new_accuracy": best_acc_for_pos,
            "accepted": best_snr_for_pos > best_snr,
            "context_Index_list": [ctx["Index"] for ctx in best_candidate_for_pos] if best_candidate_for_pos else [ctx["Index"] for ctx in current_context],
            "num_candidates_tried": len(sampled_indices)
        }
        search_history.append(result)

        # Accept if improved (using SNR)
        if best_candidate_for_pos is not None and best_snr_for_pos > best_snr:
            snr_improvement = best_snr_for_pos - best_snr
            print(f"\n✓ Iteration {iteration}: IMPROVED SNR {best_snr:.6f} -> {best_snr_for_pos:.6f} (+{snr_improvement:.6f})")
            print(f"  Position {replace_pos}: best of {len(sampled_indices)} samples was test_idx={best_sample_idx}")
            print(f"  Accuracy: {best_accuracy:.6f} -> {best_acc_for_pos:.6f}")
            best_snr = best_snr_for_pos
            best_accuracy = best_acc_for_pos
            best_context = best_candidate_for_pos
            current_context = best_candidate_for_pos
            pbar.set_postfix({"Best SNR": f"{best_snr:.6f}", "Improvements": sum(1 for r in search_history if r["accepted"])})
        else:
            pbar.set_postfix({"Best SNR": f"{best_snr:.6f}", "Improvements": sum(1 for r in search_history if r["accepted"])})

    # Save results
    print("\n=== Saving Results ===")

    # Save search history
    history_path = os.path.join(output_dir, "search_history.json")
    with open(history_path, 'w') as f:
        json.dump(search_history, f, indent=2)
    print(f"Search history saved to {history_path}")

    # Save best context
    best_context_path = os.path.join(output_dir, "refined_context.json")
    with open(best_context_path, 'w') as f:
        json.dump([{k: v for k, v in item.items() if k != "embeddings"} for item in best_context], f, indent=2)
    print(f"Best refined context saved to {best_context_path}")

    # Log preference type breakdown
    true_pref_count = sum(1 for ctx in best_context if ctx.get("is_true_preference", False))
    synth_pref_count = sum(1 for ctx in best_context if not ctx.get("is_true_preference", True))
    print(f"Final context composition: {true_pref_count} true preferences, {synth_pref_count} synthesized preferences")

    # Save best context with embeddings
    best_context_full_path = os.path.join(output_dir, "refined_context_with_embeddings.pkl")
    with open(best_context_full_path, 'wb') as f:
        pickle.dump(best_context, f)
    print(f"Best refined context with embeddings saved to {best_context_full_path}")

    # Save summary
    summary = {
        "initial_snr": initial_snr,
        "final_snr": best_snr,
        "snr_improvement": best_snr - initial_snr,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": best_accuracy,
        "accuracy_improvement": best_accuracy - initial_accuracy,
        "num_iterations": num_iterations,
        "total_accepted": sum(1 for r in search_history if r["accepted"]),
        "best_context_indices": [ctx["Index"] for ctx in best_context]
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("✓ Context refinement complete!")
    print("=" * 70)
    print(f"\nInitial SNR:     {initial_snr:.6f}")
    print(f"Final SNR:       {best_snr:.6f}")
    print(f"SNR Improvement: {best_snr - initial_snr:.6f}")
    print(f"\nInitial accuracy: {initial_accuracy:.6f}")
    print(f"Final accuracy:  {best_accuracy:.6f}")
    print(f"Accuracy improvement: {best_accuracy - initial_accuracy:.6f}")
    print(f"Iterations with improvement: {sum(1 for r in search_history if r['accepted'])}/{num_iterations}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTo use this refined context for inference:")
    print(f"  Copy {best_context_full_path}")
    print(f"  to {context_optimization_dir}/best_context_with_embeddings.pkl")


if __name__ == "__main__":
    main()
