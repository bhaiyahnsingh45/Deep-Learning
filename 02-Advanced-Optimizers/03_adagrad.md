# AdaGrad (Adaptive Gradient)

**Date:** 2026-04-26

---

## What is AdaGrad?

All optimizers so far (SGD, Momentum, NAG) use the **same learning rate α for all parameters**. This is a problem:
- Some parameters may need large updates (rare features, small gradients)
- Others need small updates (frequent features, large gradients)

**AdaGrad** solves this by giving each parameter its own **adaptive learning rate** — parameters with large historical gradients get smaller updates, and parameters with small historical gradients get larger updates.

---

## Intuition

"If a parameter has been updated a lot (large accumulated gradients), slow it down. If a parameter has barely been updated (small accumulated gradients), speed it up."

Particularly useful for **sparse data** (e.g., NLP word embeddings — most words appear rarely, a few appear constantly).

---

## Mathematics

Accumulate the sum of squared gradients for each parameter:

```
G  = G  +  (∂L/∂θ)²          ← accumulate squared gradients
θ  = θ  -  (α / √(G + ε)) * ∂L/∂θ
```

- **G** — sum of all squared gradients so far (per parameter, starts at 0)
- **ε** — small constant for numerical stability (e.g., 1e-8), prevents division by zero
- **α** — global learning rate (typically 0.01)
- Effective learning rate = **α / √G** — shrinks over time as G grows

---

## Advantages

- **Per-parameter adaptive learning rates** — no manual tuning per parameter
- **Great for sparse data** — rare parameters get larger updates automatically
- **Eliminates need to manually decay learning rate** — learning rate adapts on its own
- **Works well for NLP tasks** — word embeddings have very sparse gradient updates

## Disadvantages

- **Learning rate monotonically decreases** — G only grows, never shrinks, so effective LR keeps getting smaller
- **Learning rate eventually → 0** — training can stop making progress entirely (stuck)
- **Not suitable for non-convex deep networks** — learning rate dies too fast in long training runs
- **Accumulates all past gradients equally** — old gradients from early training affect current updates forever

---

## When to Use

- **Sparse data problems** — NLP (word embeddings), recommender systems
- **Convex problems** with sparse features
- **Avoid for deep learning** with long training — learning rate will die; use RMSProp or Adam instead

---

## Key Takeaway

AdaGrad was the first major adaptive optimizer. Its core idea — "divide by accumulated gradient magnitude per parameter" — is powerful, but the unbounded accumulation causes learning to stop. RMSProp and Adam fix this limitation.
