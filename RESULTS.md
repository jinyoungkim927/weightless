# True Everything — Results

Branch: `modification/true-everything`

## What This Branch Adds

This branch combines **ALL** modifications that were individually tested, including two that were missing from the original `everything-optimized`:

| Modification | Flag(s) | Source Branch | In everything-optimized? |
|---|---|---|---|
| ReLU² FFN | `--ffn_type relu2`, d_ff=3072 | modification/relu2-ffn | Yes |
| Muon optimizer | `--use_muon` | modification/muon-optimizer | Yes |
| QK normalization | `--qk_norm` | modification/attn-residual-tricks | Yes |
| Logit softcapping | `--softcap 15.0` | modification/attn-residual-tricks | Yes |
| Residual scalars | `--resid_scalars` | modification/attn-residual-tricks | Yes |
| QA data augmentation | `--qa_dir`, `--qa_ratio 0.1` | modification/data-optimization | Yes |
| **Weight sparsity** | `--weight_sparsity 0.3` | modification/weight-sparsity | **NEW** |
| **Copy gate** | Phase B training | modification/copy-gate | **NEW** |

## Results (50M tokens, 1 GPU)

| Run | QA Data | Val Loss | Sparsity | Story QA Acc |
|-----|---------|----------|----------|-------------|
| true_everything_50M | Yes | 3.9465 | 28.9% | **57%** |
| true_everything_no_qa_50M | No | 3.9429 | 28.9% | **15%** |

### Copy Gate Training (Phase B)

| Run | Val Loss Before | Val Loss After | Delta | p_copy |
|-----|----------------|---------------|-------|--------|
| With QA | 3.9376 | 3.9247 | -0.013 | 3.5% |
| Without QA | 3.9341 | 3.9211 | -0.013 | 3.5% |

## Comparison with everything_optimized

| Run | Modifications | Tokens | Story QA |
|-----|---------------|--------|----------|
| everything_optimized_5500_rerun | All arch mods + QA data | 0.72B | **72%** |
| everything_50M_rerun | All arch mods + QA data | 0.05B | **69%** |
| true_everything_50M | + sparsity + copy gate | 0.05B | 57% |
| true_everything_no_qa_50M | + sparsity + copy gate, NO QA | 0.05B | 15% |

## Key Findings

1. **QA data is the dominant factor**: Removing it drops accuracy from 57% to 15%
2. **Sparsity + copy gate hurt at 50M scale**: 69% → 57% when adding these modifications on top of everything-optimized
3. **Copy gate provides modest val_loss improvement** (~0.013) but doesn't significantly affect Story QA
4. Weight sparsity converged to ~28.9% (target 30%) with 48 managed parameter matrices

## WandB Links

- true_everything_50M: https://wandb.ai/jinyoungkim927/weightless/runs/t50rm09f
- true_everything_no_qa_50M: https://wandb.ai/jinyoungkim927/weightless/runs/kr80al8r
