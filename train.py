"""Training script with wandb logging and optional DDP support.

True-everything: combines Muon optimizer, ReLU² FFN, QK norm,
logit softcapping, per-layer residual scalars, QA data augmentation,
weight sparsity (RigL-style prune-regrow), and copy gate.
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Peak TFLOPS for MFU calculation (BF16 tensor core ops)
# H100 SXM: 990 TFLOPS BF16, A100: 312 TFLOPS BF16
GPU_PEAK_TFLOPS = 990

from data import get_dataloader
from model import create_model

PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|> used as pad token
GOAL_VAL_LOSS = 3.5


# ---------------------------------------------------------------------------
# DDP helpers (optional -- works without torchrun)
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (local_rank, world_size, use_ddp)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, False
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F811
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), True


def is_main(use_ddp: bool = False):
    if not use_ddp:
        return True
    import torch.distributed as dist
    return dist.get_rank() == 0


def get_world_size(use_ddp: bool = False):
    if not use_ddp:
        return 1
    import torch.distributed as dist
    return dist.get_world_size()


def get_rank(use_ddp: bool = False):
    if not use_ddp:
        return 0
    import torch.distributed as dist
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Weight sparsity: Gradual Magnitude Pruning with Gradient-Guided Regrowth
# ---------------------------------------------------------------------------

class SparseTopologyManager:
    """Manages dynamic sparse training for 2D weight matrices.

    Maintains binary masks on weight matrices. Every `update_interval` steps:
      1. Prune: zero the smallest-magnitude active weights (fraction = drop_fraction)
      2. Regrow: activate the highest-gradient dormant weights (same count)
    Total sparsity stays constant; the topology adapts to where gradients demand capacity.

    Sparsity ramps up on a cubic schedule during a warmup phase, then the topology
    continues to evolve at fixed sparsity via prune-regrow cycles.
    """

    def __init__(self, model, target_sparsity: float = 0.3,
                 warmup_steps: int = 1000, update_interval: int = 100,
                 drop_fraction: float = 0.3):
        self.target_sparsity = target_sparsity
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.drop_fraction = drop_fraction

        # Collect 2D weight matrices (skip embeddings, LayerNorm, biases, copy_gate)
        self.params = []
        self.masks = {}
        for name, param in model.named_parameters():
            if (param.ndim == 2
                    and 'token_emb' not in name
                    and 'head.' not in name
                    and 'copy_gate' not in name):
                self.params.append((name, param))
                # Start fully dense (all ones)
                self.masks[name] = torch.ones_like(param, dtype=torch.bool)

    def _current_sparsity(self, step: int) -> float:
        """Cubic ramp from 0 to target_sparsity over warmup_steps."""
        if step >= self.warmup_steps:
            return self.target_sparsity
        frac = step / self.warmup_steps
        return self.target_sparsity * (1 - (1 - frac) ** 3)

    @torch.no_grad()
    def step(self, step: int):
        """Called every training step. Only does work at update boundaries."""
        if step % self.update_interval != 0:
            # Just enforce masks every step (weights may have been updated by optimizer)
            self._apply_masks()
            return

        current_sparsity = self._current_sparsity(step)
        if current_sparsity <= 0:
            return

        for name, param in self.params:
            mask = self.masks[name]
            n_total = param.numel()
            n_zeros_target = int(current_sparsity * n_total)

            if step < self.warmup_steps:
                # During warmup: simple magnitude pruning to reach current sparsity
                magnitudes = param.abs()
                if n_zeros_target > 0:
                    threshold = torch.kthvalue(magnitudes.view(-1), n_zeros_target).values
                    new_mask = magnitudes > threshold
                    # Ensure we don't prune more than target
                    if (~new_mask).sum() > n_zeros_target:
                        new_mask = magnitudes >= threshold
                    self.masks[name] = new_mask
                    param.mul_(new_mask)
            else:
                # Post-warmup: prune-regrow cycle at fixed sparsity
                n_active = mask.sum().item()
                n_to_drop = int(self.drop_fraction * n_active)
                if n_to_drop == 0:
                    continue

                # PRUNE: remove smallest-magnitude active weights
                active_magnitudes = param.abs() * mask.float()
                active_vals = active_magnitudes[mask]
                if len(active_vals) <= n_to_drop:
                    continue
                drop_threshold = torch.kthvalue(active_vals, n_to_drop).values
                prune_mask = (active_magnitudes <= drop_threshold) & mask
                mask[prune_mask] = False

                # REGROW: activate highest-gradient dormant weights
                if param.grad is not None:
                    dormant = ~mask
                    grad_magnitudes = param.grad.abs() * dormant.float()
                    n_dormant = dormant.sum().item()
                    n_to_grow = min(n_to_drop, int(n_dormant))
                    if n_to_grow > 0:
                        grow_threshold = torch.kthvalue(
                            grad_magnitudes[dormant],
                            max(1, int(n_dormant) - n_to_grow)
                        ).values
                        grow_mask = (grad_magnitudes >= grow_threshold) & dormant
                        mask[grow_mask] = True
                        # Initialize regrown weights to zero (let optimizer fill them)
                        param[grow_mask] = 0.0

                self.masks[name] = mask
                param.mul_(mask)

    def _apply_masks(self):
        """Zero out masked weights (call after optimizer.step)."""
        for name, param in self.params:
            param.data.mul_(self.masks[name])

    def get_sparsity(self) -> float:
        """Return current actual sparsity across all managed parameters."""
        total = 0
        zeros = 0
        for name, param in self.params:
            total += param.numel()
            zeros += (~self.masks[name]).sum().item()
        return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------

def setup_optimizer(model, args, use_ddp=False):
    """Set up MuonAdamW or AdamW optimizer with proper param grouping."""
    if not args.use_muon:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.3,
        )

    from muon_optim import MuonAdamW

    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # LR scaling: (d_model/768)^-0.5
    lr_scale = (raw_model.d_model / 768) ** -0.5

    # Classify parameters
    embedding_params = []
    muon_params_by_shape = {}  # shape -> list of params
    skip_names = set()

    # Identify weight-tied lm_head
    if hasattr(raw_model, 'weight_tied') and raw_model.weight_tied:
        skip_names.add('head.weight')

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if name in skip_names:
            continue
        if 'token_emb' in name:
            embedding_params.append(param)
        elif param.ndim == 2:
            shape = param.shape
            if shape not in muon_params_by_shape:
                muon_params_by_shape[shape] = []
            muon_params_by_shape[shape].append(param)
        else:
            embedding_params.append(param)  # 1D params (layernorm etc) go to AdamW

    param_groups = []

    # Embedding / scalar params -> AdamW
    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'kind': 'adamw',
            'lr': args.embedding_lr * lr_scale,
            'betas': (0.9, 0.95),
            'eps': 1e-8,
            'weight_decay': 0.0,
        })

    # Matrix params -> Muon (one group per shape for stacking)
    for shape, params in muon_params_by_shape.items():
        param_groups.append({
            'params': params,
            'kind': 'muon',
            'lr': args.matrix_lr * lr_scale,
            'momentum': 0.85,
            'ns_steps': 5,
            'beta2': 0.7,
            'weight_decay': args.muon_wd,
        })

    return MuonAdamW(param_groups)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_loss(model, batch, device):
    """Compute cross-entropy loss for a batch using BF16 autocast."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=PAD_TOKEN_ID,
        )
    return loss


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then linear decay (used when use_muon=False)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr - (max_lr - min_lr) * decay_ratio


def get_lr_multiplier(step: int, total_steps: int) -> float:
    """Warmup (2%) -> constant (48%) -> warmdown (50%, linear to 0)."""
    warmup_steps = round(0.02 * total_steps)
    warmdown_steps = round(0.50 * total_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step <= total_steps - warmdown_steps:
        return 1.0
    else:
        progress = (total_steps - step) / warmdown_steps
        return progress


def get_muon_momentum(step: int) -> float:
    """Momentum warmup: 0.85 -> 0.95 over first 300 steps."""
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(step: int, base_wd: float, total_steps: int) -> float:
    """Linear decay of weight decay to 0 over training."""
    return base_wd * (1 - step / total_steps)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 20):
    """Evaluate model on validation set (lightweight, uses BF16)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        loss = compute_loss(model, batch, device)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_steps: int = 5000,
    eval_every: int = 50,
    max_lr: float = 1e-3,
    warmup_steps: int = 200,
    use_wandb: bool = True,
    use_ddp: bool = False,
    use_muon: bool = True,
    muon_wd: float = 0.02,
    sparse_manager: SparseTopologyManager | None = None,
):
    """Main training loop with logging every eval_every steps."""
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    num_params = raw_model.count_parameters(count_zeros=True)
    min_lr = max_lr * 0.1

    # Store base LRs for Muon schedule
    if use_muon:
        for group in optimizer.param_groups:
            group["_base_lr"] = group["lr"]

    train_iter = iter(train_loader)
    running_loss = 0.0
    total_tokens = 0
    epoch = 0
    t0 = time.time()
    world_size = get_world_size(use_ddp)

    pbar = tqdm(range(num_steps), desc="Training", disable=not is_main(use_ddp))
    for step in pbar:
        # LR scheduling
        if use_muon:
            lr_mult = get_lr_multiplier(step, num_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["_base_lr"] * lr_mult
            # Update Muon-specific per-step values
            muon_mom = get_muon_momentum(step)
            muon_wd_val = get_weight_decay(step, muon_wd, num_steps)
            for param_group in optimizer.param_groups:
                if param_group.get("kind") == "muon":
                    param_group["momentum"] = muon_mom
                    param_group["weight_decay"] = muon_wd_val
            lr = lr_mult * max_lr  # for logging
        else:
            lr = get_lr(step, warmup_steps, num_steps, max_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            epoch += 1

        B, T = batch["input_ids"].shape
        tokens_this_step = B * T * world_size
        total_tokens += tokens_this_step

        optimizer.zero_grad()
        loss = compute_loss(model, batch, device)
        loss.backward()
        optimizer.step()

        # Apply sparse topology update (prune-regrow cycle)
        if sparse_manager is not None:
            sparse_manager.step(step)

        running_loss += loss.item()

        if (step + 1) % eval_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tokens_interval = tokens_this_step * eval_every
            mfu = 6 * num_params * tokens_interval / (GPU_PEAK_TFLOPS * 1e12 * dt * world_size)

            total_flops = 6 * num_params * total_tokens
            train_loss = running_loss / eval_every
            val_loss = evaluate(model, val_loader, device)
            nonzero_params = raw_model.count_parameters(count_zeros=False)

            if is_main(use_ddp):
                pbar.set_postfix({"train": f"{train_loss:.3f}", "val": f"{val_loss:.3f}", "mfu": f"{mfu:.1%}"})
                if use_wandb:
                    import wandb
                    log_dict = {
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/total_tokens": total_tokens,
                        "train/total_flops": total_flops,
                        "train/epoch": epoch,
                        "val/loss": val_loss,
                        "params/nonzero": nonzero_params,
                        "mfu": mfu,
                        "step": step + 1,
                    }
                    if sparse_manager is not None:
                        log_dict["sparsity/weight_sparsity"] = sparse_manager.get_sparsity()
                    wandb.log(log_dict)
                if val_loss < GOAL_VAL_LOSS:
                    print(f"\n  Goal achieved! val_loss={val_loss:.4f} < {GOAL_VAL_LOSS} with {nonzero_params:,} non-zero params")

            running_loss = 0.0
            t0 = time.time()

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--num_steps", type=int, default=55000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "baseline_plus", "copy_gate"],
                        help="Model variant")
    # FFN type (from relu2-ffn)
    parser.add_argument("--ffn_type", type=str, default="relu2",
                        choices=["swiglu", "relu2"],
                        help="FFN type: swiglu (3 matrices) or relu2 (2 matrices, ReLU squared)")
    # Attention/residual tricks (from attn-residual-tricks)
    parser.add_argument("--qk_norm", action="store_true", default=True,
                        help="Enable QK normalization in attention (default: True)")
    parser.add_argument("--no_qk_norm", action="store_true",
                        help="Disable QK normalization")
    parser.add_argument("--softcap", type=float, default=15.0,
                        help="Logit softcapping value (0 to disable)")
    parser.add_argument("--resid_scalars", action="store_true", default=True,
                        help="Enable per-layer residual scalars (default: True)")
    parser.add_argument("--no_resid_scalars", action="store_true",
                        help="Disable per-layer residual scalars")
    # Muon optimizer (from muon-optimizer)
    parser.add_argument("--use_muon", action=argparse.BooleanOptionalAction, default=True,
                        help="Use MuonAdamW optimizer (default: True)")
    parser.add_argument("--matrix_lr", type=float, default=0.005,
                        help="Muon learning rate for 2D matrix params")
    parser.add_argument("--embedding_lr", type=float, default=0.05,
                        help="AdamW learning rate for embeddings")
    parser.add_argument("--muon_wd", type=float, default=0.005,
                        help="Muon weight decay (linearly decayed to 0)")
    # Data augmentation (from data-optimization)
    parser.add_argument("--qa_dir", type=str, default=None,
                        help="Path to QA-augmented parquet directory for data mixing")
    parser.add_argument("--qa_ratio", type=float, default=0.1,
                        help="Fraction of QA samples when --qa_dir is set")
    # Weight sparsity (RigL-style prune-regrow)
    parser.add_argument("--weight_sparsity", type=float, default=0.0,
                        help="Target weight sparsity (0 to disable)")
    parser.add_argument("--sparsity_warmup", type=int, default=1000,
                        help="Steps to ramp sparsity from 0 to target")
    parser.add_argument("--prune_interval", type=int, default=100,
                        help="Steps between prune-regrow cycles")
    parser.add_argument("--drop_fraction", type=float, default=0.3,
                        help="Fraction of active weights to prune/regrow each cycle")
    # Logging and tracking
    parser.add_argument("--wandb_project", type=str, default="weightless")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (used for checkpoint filename)")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Save model checkpoint at end of training")
    parser.add_argument("--modification", type=str, default=None,
                        help="Description of modification for experiment tracker")
    parser.add_argument("--intuition", type=str, default=None,
                        help="Why this modification was made (for experiment tracker)")
    parser.add_argument("--no_auto_eval", action="store_true",
                        help="Skip automatic post-training evaluation")
    args = parser.parse_args()

    # Process attention/residual trick flags
    if args.no_qk_norm:
        args.qk_norm = False
    if args.no_resid_scalars:
        args.resid_scalars = False

    # Autoscale max_lr: args.max_lr is calibrated at d_model=768
    # (Only for AdamW fallback - Muon handles its own scaling)
    if not args.use_muon:
        base_lr = args.max_lr
        args.max_lr = base_lr * (768 / args.d_model) ** 0.5

    # DDP setup (optional -- works with plain `python train.py`)
    local_rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    use_wandb = not args.no_wandb

    if is_main(use_ddp):
        if args.use_muon:
            print(f"  Muon optimizer: matrix_lr={args.matrix_lr}, embedding_lr={args.embedding_lr}, muon_wd={args.muon_wd}")
        else:
            if args.d_model != 768:
                print(f"  Autoscaled max_lr from {base_lr:.2e} to {args.max_lr:.2e} (d_model={args.d_model})")
            else:
                print(f"  max_lr={args.max_lr:.2e} (d_model={args.d_model})")
        if use_ddp:
            print(f"  DDP: rank {local_rank}, world_size {world_size}")
        else:
            print(f"  Single-GPU mode (use torchrun for DDP)")

    # Enable flash attention and bf16 optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # Run name
    if args.run_name is None:
        args.run_name = f"{args.model}_d{args.d_model}_L{args.n_layers}"

    # wandb
    if use_wandb and is_main(use_ddp):
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={**vars(args), "model_variant": args.model},
        )

    # Data
    rank = get_rank(use_ddp)
    if is_main(use_ddp):
        print("  Setting up data loaders...")
    train_loader = get_dataloader(split="train", batch_size=args.batch_size,
                                  streaming=True, rank=rank, world_size=world_size,
                                  qa_dir=args.qa_dir, qa_ratio=args.qa_ratio)
    val_loader = get_dataloader(split="test", batch_size=args.batch_size,
                                streaming=True, rank=rank, world_size=world_size)

    # Model
    if is_main(use_ddp):
        print(f"  Creating model (variant={args.model}, BF16 + torch.compile)...")
    model = create_model(
        variant=args.model,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        ffn_type=args.ffn_type,
        qk_norm=args.qk_norm,
        softcap=args.softcap,
        use_resid_scalars=args.resid_scalars,
    )
    model.to(device)
    model = torch.compile(model)

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if hasattr(model, "module") else model
    # Handle torch.compile wrapper
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    total_params = raw_model.count_parameters(count_zeros=True)
    nonzero_params = raw_model.count_parameters(count_zeros=False)

    if is_main(use_ddp):
        print(f"  Total parameters:    {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")

        if use_wandb:
            import wandb
            wandb.log({
                "params/total": total_params,
                "params/nonzero": nonzero_params,
            })

    # Optimizer
    optimizer = setup_optimizer(model, args, use_ddp=use_ddp)

    # Weight sparsity manager (operates on raw model params)
    sparse_manager = None
    if args.weight_sparsity > 0:
        sparse_manager = SparseTopologyManager(
            raw_model,
            target_sparsity=args.weight_sparsity,
            warmup_steps=args.sparsity_warmup,
            update_interval=args.prune_interval,
            drop_fraction=args.drop_fraction,
        )
        if is_main(use_ddp):
            print(f"  Weight sparsity: target={args.weight_sparsity}, "
                  f"warmup={args.sparsity_warmup}, interval={args.prune_interval}, "
                  f"drop_fraction={args.drop_fraction}")
            print(f"  Sparsity-managed params: {len(sparse_manager.params)} matrices")

    # Train
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_lr=args.max_lr,
        use_wandb=use_wandb,
        use_ddp=use_ddp,
        use_muon=args.use_muon,
        muon_wd=args.muon_wd,
        sparse_manager=sparse_manager,
    )

    # End-of-training summary
    if is_main(use_ddp):
        raw_model_final = model.module if hasattr(model, "module") else model
        if hasattr(raw_model_final, "_orig_mod"):
            raw_model_final = raw_model_final._orig_mod
        final_val_loss = evaluate(model, val_loader, device)
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Final val_loss:         {final_val_loss:.4f}  (target < {GOAL_VAL_LOSS})")
        if final_val_loss < GOAL_VAL_LOSS:
            print(f"  GOAL ACHIEVED!")
        else:
            print(f"  Goal not yet reached (val_loss {final_val_loss:.4f} >= {GOAL_VAL_LOSS})")
        print()

        # Save checkpoint
        if args.save_checkpoint:
            import os
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/{args.run_name}.pt"
            torch.save(raw_model_final.state_dict(), ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    # Auto-run experiment tracker after training (rank 0 only)
    if is_main(use_ddp) and args.save_checkpoint and not args.no_auto_eval:
        wandb_url = ""
        if use_wandb:
            import wandb as _wb
            if _wb.run is not None:
                wandb_url = _wb.run.get_url() or ""

        _run_post_training_eval(args, wandb_url)

    if use_wandb and is_main(use_ddp):
        import wandb
        wandb.finish()
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


def _run_post_training_eval(args, wandb_url: str = ""):
    """Run experiment tracker pipeline after training completes."""
    try:
        from experiment_tracker import run_full_pipeline
    except ImportError:
        print("  experiment_tracker.py not found, skipping post-training eval")
        return

    ckpt_path = f"checkpoints/{args.run_name}.pt"
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found at {ckpt_path}, skipping post-training eval")
        return

    # Auto-generate modification description from flags if not provided
    modification = args.modification
    if not modification:
        parts = []
        if getattr(args, 'qk_norm', False):
            parts.append("QK normalization")
        if getattr(args, 'softcap', 0) > 0:
            parts.append(f"logit softcapping ({args.softcap})")
        if getattr(args, 'resid_scalars', False):
            parts.append("per-layer residual scalars")
        if getattr(args, 'use_muon', False):
            parts.append("Muon optimizer")
        if getattr(args, 'ffn_type', 'swiglu') == 'relu2':
            parts.append(f"ReLU2 FFN (d_ff={args.d_ff})")
        modification = " + ".join(parts) if parts else "baseline (no modifications)"

    intuition = args.intuition or "See branch documentation"

    # Build log path
    log_path = f"train_{args.run_name}.log"
    if not os.path.exists(log_path):
        log_path = None

    # Build extra model kwargs for non-baseline architectures
    extra_model_kwargs = {}
    if getattr(args, 'ffn_type', None) and args.ffn_type != 'swiglu':
        extra_model_kwargs["ffn_type"] = args.ffn_type
    if getattr(args, 'qk_norm', False):
        extra_model_kwargs["qk_norm"] = True
    if getattr(args, 'softcap', 0) > 0:
        extra_model_kwargs["softcap"] = args.softcap
    if getattr(args, 'resid_scalars', False):
        extra_model_kwargs["use_resid_scalars"] = True

    print(f"\n{'='*60}")
    print(f"  AUTO POST-TRAINING EVALUATION")
    print(f"  Modification: {modification}")
    print(f"{'='*60}")

    try:
        run_full_pipeline(
            run_name=args.run_name,
            modification=modification,
            intuition=intuition,
            checkpoint_path=ckpt_path,
            model_variant=args.model,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            wandb_url=wandb_url,
            log_path=log_path,
            extra_model_kwargs=extra_model_kwargs or None,
        )
    except Exception as e:
        print(f"\n  Post-training eval failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Training was successful -- checkpoint was saved. Run experiment_tracker.py manually.")


if __name__ == "__main__":
    main()
