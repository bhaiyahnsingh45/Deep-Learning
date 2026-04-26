# Gradient Descent

**Date:** 2026-04-26

---

## What is Gradient Descent?

Gradient Descent is an **optimization algorithm** used to minimize a loss function by iteratively moving in the direction of the **steepest descent** (negative gradient).

In deep learning, we want to find the model parameters (weights **W** and biases **b**) that minimize the loss function **L**.

---

## The Core Idea — Intuition

Imagine you are standing on a hilly landscape (the loss surface) and you want to reach the lowest valley (minimum loss). At each step you:

1. Look around and find which direction is steepest downhill (compute the gradient)
2. Take a step in that direction
3. Repeat until you reach a valley

The **learning rate** (α) controls how big each step is.

---

## Mathematics

### Gradient

The gradient of the loss with respect to a parameter θ:

```
∂L/∂θ
```

It tells us: "if I increase θ slightly, how much does the loss increase?"

### Parameter Update Rule

```
θ = θ - α * ∂L/∂θ
```

- **θ** — model parameter (weight or bias)
- **α** — learning rate (hyperparameter, e.g., 0.01)
- **∂L/∂θ** — gradient of loss w.r.t. parameter

---

## The Three Variants

---

### 1. Batch Gradient Descent (BGD)

**Idea:** Compute the gradient using the **entire training dataset** before making one update.

```
θ = θ - α * (1/m) * Σ ∂L(xᵢ, yᵢ)/∂θ    [sum over all m samples]
```

**Pros:**
- Stable, smooth convergence
- Guaranteed to converge to global minimum for convex functions

**Cons:**
- Very slow for large datasets (one update per full pass)
- Requires entire dataset in memory
- Computationally expensive per step

**When to use:** Small datasets where accuracy of gradient matters more than speed.

---

### 2. Stochastic Gradient Descent (SGD)

**Idea:** Compute the gradient using **one random training sample** at a time, update immediately.

```
θ = θ - α * ∂L(xᵢ, yᵢ)/∂θ    [single sample]
```

**Pros:**
- Much faster updates — can start learning immediately
- Can escape local minima due to noisy updates
- Works well with online learning (streaming data)

**Cons:**
- High variance in updates — loss curve is very noisy
- May never fully converge, keeps oscillating around minimum
- Harder to parallelize

**When to use:** Very large datasets, online learning scenarios.

---

### 3. Mini-Batch Gradient Descent

**Idea:** Compute the gradient using a **small batch** of B samples (typically 32, 64, 128, 256).

```
θ = θ - α * (1/B) * Σ ∂L(xᵢ, yᵢ)/∂θ    [sum over batch of size B]
```

**Pros:**
- Balance between BGD and SGD — less noisy than SGD, faster than BGD
- Leverages vectorized operations (GPU-friendly)
- Most widely used in practice

**Cons:**
- One more hyperparameter to tune (batch size)
- Gradient is still an approximation

**When to use:** This is the **default choice** in modern deep learning.

---

## Comparison Table

| Property | Batch GD | SGD | Mini-Batch GD |
|----------|:--------:|:---:|:-------------:|
| Samples per update | All (m) | 1 | B (32–256) |
| Updates per epoch | 1 | m | m/B |
| Gradient accuracy | Exact | Noisy (high variance) | Approximate |
| Loss curve shape | Smooth & monotone | Very noisy | Slightly noisy |
| Memory per step | High (all data) | Low (1 sample) | Medium (B samples) |
| GPU efficiency | Low | Very low | **High** |
| Convergence speed | Slow | Fast early on | **Fast + stable** |
| Can escape local minima | No | Yes (noise helps) | Somewhat |
| Supports online learning | No | Yes | No |
| Typical use in practice | Rarely | Sometimes | **Default choice** |

### Quick Rule of Thumb

```
Small dataset  (<10K)  → Batch GD
Streaming data         → SGD
Everything else        → Mini-Batch GD  (batch_size = 32 or 64)
```

---

## Learning Rate — Key Hyperparameter

| Learning Rate | Effect |
|--------------|--------|
| Too large (α = 1.0) | Overshoots, diverges |
| Too small (α = 0.0001) | Converges very slowly |
| Just right (α = 0.01) | Smooth convergence |

---

## Challenges with Gradient Descent

1. **Choosing learning rate** — too big diverges, too small is slow
2. **Saddle points** — gradient is zero but not a minimum
3. **Local minima** — getting stuck (less of a problem in high-dim spaces)
4. **Vanishing/exploding gradients** — gradients become too small or too large
5. **Plateau regions** — flat areas where gradient is near zero

These challenges motivate advanced optimizers: Momentum, RMSProp, Adam (covered later).

---

## Key Takeaways

- Gradient Descent updates parameters in the direction that reduces loss
- **Batch GD** = stable but slow; **SGD** = fast but noisy; **Mini-Batch** = best of both
- Mini-Batch GD with a batch size of 32–256 is the default in deep learning
- Learning rate is the most important hyperparameter to tune

---

**Next Topic:** Perceptron & Activation Functions
