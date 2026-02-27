#!/usr/bin/env python3
"""Lightweight training script for the copy gate mechanism.

Loads a frozen baseline checkpoint and trains only the copy_gate parameters
(~769 params: 768 weights + 1 bias). Uses standard LM cross-entropy loss.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_copy_gate.py
    CUDA_VISIBLE_DEVICES=0 python train_copy_gate.py --checkpoint checkpoints/baseline_full_8gpu.pt --num_steps 1000
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import get_dataloader
from model import create_model

PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|> used as pad token


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=20):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train copy gate on frozen baseline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_full_8gpu.pt",
                        help="Path to baseline checkpoint")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--ffn_type", type=str, default="relu2",
                        choices=["swiglu", "relu2"])
    parser.add_argument("--qk_norm", action="store_true", default=True)
    parser.add_argument("--softcap", type=float, default=15.0)
    parser.add_argument("--resid_scalars", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="checkpoints/copy_gate.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Create copy_gate model
    print("  Creating CopyGateTransformer …")
    model = create_model(
        variant="copy_gate",
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        ffn_type=args.ffn_type,
        qk_norm=args.qk_norm,
        softcap=args.softcap,
        use_resid_scalars=args.resid_scalars,
    )

    # Load baseline weights (strict=False ignores missing copy_gate params)
    if os.path.exists(args.checkpoint):
        print(f"  Loading baseline checkpoint: {args.checkpoint}")
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Missing keys (expected): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    else:
        print(f"  WARNING: checkpoint not found at {args.checkpoint}, using random init!")

    model.to(device)

    # Freeze everything, unfreeze only copy_gate
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = 0
    for name, param in model.named_parameters():
        if "copy_gate" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"  Trainable: {name} ({param.numel()} params)")

    print(f"  Total trainable parameters: {trainable_params}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total model parameters: {total_params:,}")

    # Optimizer (only gate params)
    gate_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(gate_params, lr=args.lr, weight_decay=0.0)

    # Data
    print("  Setting up data loaders …")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size, streaming=True)
    val_loader = get_dataloader(split="test", batch_size=args.batch_size, streaming=True)

    # Initial eval
    print("  Evaluating baseline (before copy gate training) …")
    val_loss_init = evaluate(model, val_loader, device)
    print(f"  Initial val_loss: {val_loss_init:.4f}")

    # Training loop
    print(f"\n  Training copy gate for {args.num_steps} steps …")
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    best_val_loss = val_loss_init
    t0 = time.time()

    pbar = tqdm(range(args.num_steps), desc="Copy Gate Training")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=PAD_TOKEN_ID,
            )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            # Check gate values
            with torch.no_grad():
                gate_weight = model.copy_gate.linear.weight
                gate_bias = model.copy_gate.linear.bias
                # p_copy at bias point (when h=0): sigmoid(bias)
                p_copy_base = torch.sigmoid(gate_bias).item()

            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "p_copy_base": f"{p_copy_base:.3f}",
                "elapsed": f"{elapsed:.0f}s",
            })
            print(f"\n  Step {step+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, p_copy_base={p_copy_base:.3f}, "
                  f"gate_weight_norm={gate_weight.norm().item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best val_loss: {val_loss:.4f}")

            running_loss = 0.0
            t0 = time.time()

    # Final eval
    final_val_loss = evaluate(model, val_loader, device)
    print(f"\n{'='*60}")
    print(f"  COPY GATE TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Initial val_loss:  {val_loss_init:.4f}")
    print(f"  Final val_loss:    {final_val_loss:.4f}")
    print(f"  Best val_loss:     {best_val_loss:.4f}")
    print(f"  Delta:             {final_val_loss - val_loss_init:+.4f}")

    # Save checkpoint (full model state)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"  Checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
