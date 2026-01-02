#!/usr/bin/env python3
"""
Inference script for VAE preference model using demo+validation context.

This script:
1. Loads survey data from survey_100.jsonl
2. Splits into 10 demo and 10 validation samples
3. Combines all 20 samples as fixed context
4. Applies combined context to test data
5. Runs inference on test set using the trained VAE model
6. Saves predictions and metrics
"""

import os
import json
import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

# Add path for imports
# sys.path.insert(0, '/hpc/group/fanglab/xx102/vpl_llm/hidden_context')

from utils import (
    load_demo_from_survey_and_validation_from_train,
    ScriptArguments,
    RewardDataCollatorWithPadding,
    load_checkpoint,
    preprocess_validation_dataset,
    apply_context_to_dataset,
    move_batch_to_device,
    load_split_dataset,
    save_jsonl,
)
from inference import run_inference


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference with demo+validation context")
    parser.add_argument(
        "--subset",
        type=str,
        default="8",
        choices=["1", "2", "4", "8"],
        help="Data subset to evaluate: '1', '2', '4', or '8'. Default: 8"
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
        help="Number of demo examples to use from survey. Default: 50"
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=50,
        help="Number of validation examples to use from train. Default: 50"
    )
    cmd_args = parser.parse_args()

    # Configuration
    data_subset = cmd_args.subset
    seed = cmd_args.seed
    demo_size = cmd_args.demo_size
    validation_size = cmd_args.validation_size

    survey_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/UltraFeedback_single_P_4"
    train_data_path = "/hpc/group/fanglab/xx102/vpl_llm/data/data_release/P_4_survey_100/gpt2"
    checkpoint_path = "/hpc/group/fanglab/xx102/vpl_llm/logs/gpt2_P_4_survey_100/all/vae_gpt2__0_0.0001_cosine_2_3e-06_512_768_seed0_peft_last_checkpoint"

    # Create results directory structure with all arguments in folder name
    results_base_dir = "/hpc/group/fanglab/xx102/vpl_llm/results"
    output_dir = os.path.join(
        results_base_dir,
        f"inference_dem_val_subset_{data_subset}_demo_{demo_size}_validation_{validation_size}_seed{seed}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Data subset: {data_subset}")
    print(f"Demo size: {demo_size}")
    print(f"Validation size: {validation_size}")
    print(f"Random seed: {seed}")
    print(f"Context: DEMO ({demo_size}) + VALIDATION ({validation_size}) = {demo_size + validation_size} samples")

    # Setup arguments - matching training script arguments
    script_args = ScriptArguments()
    script_args.max_length = 1024
    script_args.per_device_eval_batch_size = 1
    script_args.fixed_contexts = True
    script_args.fixed_llm_embeddings = False
    script_args.other_subsets = "single"
    script_args.seed = seed

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

    # Combine all demo and validation data as context
    combined_context = demo_data + validation_data
    print(f"\n=== Combined Context ===")
    print(f"Total context samples: {len(combined_context)} (demo + validation)")

    # Load test data
    print("\n=== Loading Test Data ===")
    test_dataset = load_split_dataset(train_data_path, split="test", subset=data_subset)
    print(f"Loaded {len(test_dataset)} test samples")

    # Apply combined context to test data
    print("\n=== Applying Demo+Validation Context to Test Set ===")
    test_data_with_context = apply_context_to_dataset(list(test_dataset), combined_context)
    print(f"Using {len(combined_context)} context samples for inference")

    # Preprocess dataset matching training configuration
    print("\n=== Preprocessing Dataset ===")
    test_dataset = preprocess_validation_dataset(test_data_with_context, tokenizer, script_args)
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
    print("\n=== Running Inference with Demo+Validation Context ===")
    metrics, predictions = run_inference(vae_model, test_dataloader, device=device)

    # Print results
    print("\n=== Inference Results with Demo+Validation Context ===")
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

    # Save context info
    context_info_path = os.path.join(output_dir, "context_info.json")
    with open(context_info_path, 'w') as f:
        json.dump([{k: v for k, v in item.items() if k != "embeddings"} for item in combined_context], f, indent=2)
    print(f"Context info saved to {context_info_path}")

    # Create summary report
    report_path = os.path.join(output_dir, "inference_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VAE Preference Model Inference Report (Demo+Validation Context)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Training Data Path: {train_data_path}\n")
        f.write(f"Survey Data Path: {survey_data_path}\n")
        f.write(f"Data Subset: {data_subset}\n")
        f.write(f"Context: Demo ({demo_size}) + Validation ({validation_size}) = {len(combined_context)} samples\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Test Samples Evaluated: {metrics['num_samples']}\n\n")
        f.write("Metrics:\n")
        f.write("-" * 70 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<50} {value:.6f}\n")
            else:
                f.write(f"{key:.<50} {value}\n")
    print(f"Report saved to {report_path}")

    print("\n✓ Inference complete!")
    print(f"Output directory: {output_dir}")
    print(f"Context samples: {len(combined_context)}")
    print(f"Test samples evaluated: {metrics['num_samples']}")


if __name__ == "__main__":
    main()
