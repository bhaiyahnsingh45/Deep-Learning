# Advanced Optimizers — Overview & Comparison

**Date:** 2026-04-26

---

## The Problem with Vanilla SGD

Vanilla SGD has two main problems:
1. **Same learning rate for all parameters** — can't handle features with different frequencies
2. **Noisy, slow convergence** — especially in ravines and saddle points

The optimizers below solve one or both of these problems.

---

## Evolution of Optimizers

```
Vanilla SGD
    │
    ├── Add momentum (direction memory)
    │       ├── Momentum SGD
    │       └── NAG (smarter momentum — lookahead)
    │
    ├── Add adaptive LR (per-parameter scaling)
    │       ├── AdaGrad  (accumulate all gradients)
    │       └── RMSProp  (exponential moving avg of gradients) ← fixes AdaGrad
    │
    └── Combine momentum + adaptive LR
            └── Adam  (momentum + RMSProp + bias correction)  ← gold standard
```

---

## Full Comparison Table

| Property | SGD | Momentum | NAG | AdaGrad | RMSProp | Adam |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Adaptive LR per param | No | No | No | Yes | Yes | Yes |
| Uses momentum | No | Yes | Yes | No | No | Yes |
| Lookahead gradient | No | No | Yes | No | No | No |
| LR dies over time | No | No | No | Yes ⚠️ | No | No |
| Bias correction | No | No | No | No | No | Yes |
| Memory overhead | Low | Low | Low | Medium | Medium | High |
| Handles sparse data | Poor | Poor | Poor | Great | Good | Good |
| Good for RNNs | No | No | No | No | Yes | Yes |
| Good for Transformers | No | No | No | No | No | **Yes** |
| Default hyperparams work | No | Mostly | Mostly | Mostly | Mostly | **Yes** |
| General recommendation | Baseline | Better | Best no-adaptive | Sparse only | Good alt | **Default** |

---

## Update Rules — Side by Side

```
SGD:        θ = θ - α · g

Momentum:   v = β·v + g
            θ = θ - α·v

NAG:        v = β·v + ∇L(θ - β·v)        ← gradient at lookahead
            θ = θ - α·v

AdaGrad:    G = G + g²                    ← accumulate all squared grads
            θ = θ - (α/√(G+ε)) · g

RMSProp:    E[g²] = β·E[g²] + (1-β)·g²  ← exponential moving avg
            θ = θ - (α/√(E[g²]+ε)) · g

Adam:       m = β₁·m + (1-β₁)·g          ← 1st moment
            v = β₂·v + (1-β₂)·g²         ← 2nd moment
            m̂ = m/(1-β₁ᵗ)               ← bias correction
            v̂ = v/(1-β₂ᵗ)               ← bias correction
            θ = θ - α·m̂/(√v̂+ε)
```

Where: **g = ∂L/∂θ** (gradient), **α** = learning rate, **β** = decay coefficient, **ε** = 1e-8

---

## Recommended Defaults

| Optimizer | α | β / β₁ | β₂ |
|---|---|---|---|
| SGD | 0.01 | — | — |
| Momentum | 0.01 | 0.9 | — |
| NAG | 0.01 | 0.9 | — |
| AdaGrad | 0.01 | — | — |
| RMSProp | 0.001 | 0.9 | — |
| Adam | 0.001 | 0.9 | 0.999 |

---

## Quick Decision Guide

```
Just want something that works?          → Adam (α=0.001)
Image classification, need best accuracy → SGD + Momentum (α=0.01, β=0.9)
Sparse features (NLP embeddings)?        → AdaGrad or Adam
RNNs / LSTMs?                            → RMSProp or Adam
Transformers / LLMs?                     → AdamW (Adam + decoupled weight decay)
Research, need reproducibility?          → SGD + Momentum
```

---

## Key Insight

Every optimizer after vanilla SGD is trying to answer two questions better:
1. **Which direction?** — Momentum and NAG use gradient history to smooth direction
2. **How big a step?** — AdaGrad, RMSProp, Adam scale the step per-parameter

Adam answers both, which is why it's the default.
