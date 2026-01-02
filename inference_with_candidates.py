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
from utils import (
    load_demo_from_survey_and_validation_from_train,
    ScriptArguments,
    RewardDataCollatorWithPadding,
    load_split_dataset,
    preprocess_dataset_matching_training,
    load_checkpoint,
    create_candidate_contexts,
    apply_context_to_dataset,
)
from inference import run_inference


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
    parser.add_argument(
        "--demo-size",
        type=int,
        default=50,
        help="Number of demo examples to use. Default: 50"
    )
    args = parser.parse_args()

    # Configuration
    data_subset = args.subset
    seed = args.seed
    demo_size = args.demo_size
    num_candidates = args.num_candidates
    context_length = args.context_length

    test_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    train_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    survey_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/UltraFeedback_single_P_4"
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"

    # Create results directory structure with all arguments in folder name
    results_base_dir = "/hpc/group/fanglab/xx102/vpl_llm/results"
    output_dir = os.path.join(
        results_base_dir,
        f"candidate_evaluation_subset_{data_subset}_demo_{demo_size}_candidates_{num_candidates}_context_{context_length}_seed{seed}"
    )

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Data subset: {data_subset}")
    print(f"Demo size: {demo_size}")
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
        load_dem_only=True,
        seed=seed,
    )
    demo_data = list(demo_datasets[data_subset]) if hasattr(demo_datasets[data_subset], '__iter__') else demo_datasets[data_subset]
    print(f"Demo samples: {len(demo_data)}")

    # Load test data
    print("\n=== Loading Test Data ===")
    test_dataset = load_split_dataset(test_data_path, split="test", subset=data_subset)
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
    breakpoint()
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
