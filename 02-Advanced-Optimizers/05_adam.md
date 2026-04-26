# Adam (Adaptive Moment Estimation)

**Date:** 2026-04-26

---

## What is Adam?

**Adam** combines the best ideas from Momentum and RMSProp into one optimizer:

- From **Momentum** → tracks the exponential moving average of **gradients** (first moment = direction)
- From **RMSProp** → tracks the exponential moving average of **squared gradients** (second moment = magnitude)

It adapts the learning rate per parameter AND uses momentum — making it the most powerful and widely used optimizer in deep learning.

---

## Intuition

- **First moment (m)** — "Where have the gradients been pointing on average?" (direction)
- **Second moment (v)** — "How large have the gradients been on average?" (magnitude)
- **Bias correction** — early in training, m and v are initialized to 0, so they're biased toward 0; Adam corrects for this

---

## Mathematics

### Step 1: Compute moments

```
m  = β₁ * m  +  (1 - β₁) * ∂L/∂θ         ← 1st moment: moving avg of gradient
v  = β₂ * v  +  (1 - β₂) * (∂L/∂θ)²     ← 2nd moment: moving avg of squared gradient
```

### Step 2: Bias correction

```
m̂  = m / (1 - β₁ᵗ)     ← corrected 1st moment
v̂  = v / (1 - β₂ᵗ)     ← corrected 2nd moment
```

- **t** — current timestep (starts at 1)
- At t=1, β₁ᵗ ≈ 0.9, so (1 - β₁ᵗ) = 0.1, scaling m up to remove the initialization bias

### Step 3: Update

```
θ  = θ - α * m̂ / (√v̂ + ε)
```

### Default Hyperparameters (almost never need to change)

| Hyperparameter | Default | Meaning |
|---|---|---|
| α | 0.001 | Learning rate |
| β₁ | 0.9 | Momentum decay (1st moment) |
| β₂ | 0.999 | RMSProp decay (2nd moment) |
| ε | 1e-8 | Numerical stability |

---

## How Adam = Momentum + RMSProp

```
Momentum:   θ = θ - α * m         (uses gradient direction)
RMSProp:    θ = θ - α * g/√v      (adapts by gradient magnitude)
Adam:       θ = θ - α * m̂/√v̂    (uses both direction AND magnitude, with bias correction)
```

---

## Advantages

- **Best of both worlds** — momentum (direction) + adaptive LR (magnitude)
- **Bias correction** — accurate estimates even at the start of training
- **Per-parameter adaptive learning rate** — different parameters learn at different rates
- **Very robust** — works well across a wide range of architectures and tasks
- **Default choice** in most deep learning projects — CNNs, Transformers, LSTMs
- **Fast convergence** — usually reaches good solutions faster than SGD variants

## Disadvantages

- **Can generalize worse than SGD** — on some tasks (especially image classification), SGD+Momentum finds flatter, better-generalizing minima
- **More memory** — stores two additional tensors (m and v) per parameter
- **Sharp minima** — Adam tends to converge to sharp minima which can be less robust
- **More hyperparameters** — though defaults work well, 3 extra params (β₁, β₂, ε)
- **Not always best for fine-tuning** — SGD sometimes preferred when fine-tuning pre-trained models

---

## Adam vs SGD with Momentum — When to Choose

| Scenario | Recommended |
|---|---|
| Training from scratch, fast convergence needed | Adam |
| Image classification (ResNet, VGG) | SGD + Momentum often wins |
| NLP, Transformers, LLMs | Adam (or AdamW) |
| RNNs, LSTMs | Adam |
| Fine-tuning pre-trained models | SGD or AdamW |
| When you just want it to work | **Adam** |

---

## AdamW — Important Variant

Standard Adam applies weight decay incorrectly (mixes it with the adaptive scaling). **AdamW** fixes this by decoupling weight decay from the gradient update:

```
θ = θ - α * (m̂ / (√v̂ + ε)  +  λ * θ)     ← weight decay applied separately
```

AdamW is now the default in most modern training (especially Transformers / LLMs).

---

## Key Takeaway

Adam is the go-to optimizer for most deep learning tasks. It combines momentum (for direction) and RMSProp (for per-parameter adaptive magnitude) with bias correction. The default hyperparameters (α=0.001, β₁=0.9, β₂=0.999) work in almost every situation — it just works.
