# RMSProp (Root Mean Square Propagation)

**Date:** 2026-04-26

---

## What is RMSProp?

**RMSProp** was proposed by Geoffrey Hinton to fix AdaGrad's dying learning rate problem.

AdaGrad accumulates **all** past squared gradients (G keeps growing forever). RMSProp replaces that with an **exponentially weighted moving average** of squared gradients — so old gradients gradually fade and recent gradients matter more.

---

## Intuition

Instead of "remember every gradient since the beginning", RMSProp says "remember only the recent gradient history" using an exponential decay. This keeps the effective learning rate alive and relevant throughout training.

---

## Mathematics

```
E[g²]  =  β * E[g²]  +  (1 - β) * (∂L/∂θ)²    ← exponential moving avg of squared gradients
θ      =  θ - (α / √(E[g²] + ε)) * ∂L/∂θ
```

- **E[g²]** — exponentially weighted average of squared gradients (not a sum)
- **β** — decay rate (typically 0.9) — controls how much past gradients are remembered
- **α** — learning rate (typically 0.001)
- **ε** — small constant (1e-8) for numerical stability

### AdaGrad vs RMSProp — Core Difference

| | AdaGrad | RMSProp |
|---|---|---|
| Accumulation | Sum of all squared gradients | Exponential moving avg of squared gradients |
| Old gradients | Never forgotten | Exponentially decayed |
| Effective LR | Monotonically decreases → 0 | Stays alive |

---

## Advantages

- **Fixes AdaGrad's dying LR** — exponential decay keeps learning rate from vanishing
- **Per-parameter adaptive learning rates** — each parameter adapts independently
- **Works well for non-stationary objectives** — recent gradients get more weight
- **Good for RNNs** — handles the varying gradient scales well
- **Simple and effective** — widely used before Adam

## Disadvantages

- **No momentum** — doesn't use gradient direction history (only magnitude)
- **Learning rate still needs tuning** — α is sensitive
- **Not invariant to gradient scale** — performance can vary with different problem scales
- **Lacks bias correction** — early estimates of E[g²] are biased (Adam fixes this)
- **Generally outperformed by Adam** in most deep learning tasks

---

## When to Use

- **RNNs and LSTMs** — historically performs well here
- When AdaGrad dies too fast
- When you want adaptive LR without the full complexity of Adam
- Default **α = 0.001, β = 0.9**

---

## Key Takeaway

RMSProp kept the core idea of AdaGrad (per-parameter adaptive LR by dividing by gradient magnitude) but replaced the ever-growing sum with a moving average. This one change keeps training alive for longer. RMSProp + Momentum together inspired Adam.
