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
import argparse
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import sys
import random

# Add path for imports
sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

from vae_utils import VAEModel
from utils import (
    load_demo_from_survey_and_validation_from_train,
    ScriptArguments,
    RewardDataCollatorWithPadding,
    load_checkpoint,
    create_candidate_contexts,
    apply_context_to_dataset,
    create_demo_based_context,
    preprocess_validation_dataset,
    evaluate_context,
    save_jsonl,
)




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
    num_candidates = 50
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
