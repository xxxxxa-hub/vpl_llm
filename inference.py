#!/usr/bin/env python3
"""
Inference utilities for VAE preference model.

This module provides inference functionality for running the VAE model
on test data and computing relevant metrics.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader


def run_inference(
    model,
    test_dataloader: DataLoader,
    device: str = "cuda",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run inference on test data.

    Args:
        model: The VAE model to run inference with
        test_dataloader: DataLoader containing the test data
        device: Device to run inference on (default: "cuda")

    Returns:
        Tuple of (metrics, detailed_results) where:
        - metrics: Dictionary containing aggregated metrics
        - detailed_results: Dictionary containing per-sample results
    """
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
