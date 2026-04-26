# Momentum SGD

**Date:** 2026-04-26

---

## What is Momentum?

Vanilla SGD updates parameters using only the current gradient. The problem — it oscillates a lot and moves slowly in the right direction.

**Momentum** fixes this by keeping a running average of past gradients (called **velocity**) and using that to update parameters. Like a ball rolling downhill — it accelerates in the consistent direction and dampens oscillations.

---

## Intuition

Think of a ball rolling down a valley:
- Without momentum → ball moves only based on current slope, bounces side to side
- With momentum → ball accumulates speed in the downhill direction, ignores small bumps

---

## Mathematics

Introduce a velocity term **v**:

```
v  = β * v  +  (1 - β) * ∂L/∂θ
θ  = θ - α * v
```

Some implementations use:
```
v  = β * v  +  ∂L/∂θ
θ  = θ - α * v
```

- **v** — velocity (exponentially weighted moving average of gradients)
- **β** — momentum coefficient (typically 0.9)
- **α** — learning rate
- **β = 0** → reduces to vanilla SGD

### What β = 0.9 means
The current gradient contributes 10%, and 90% comes from the accumulated past velocity. This smooths out noise while keeping the overall trend.

---

## Advantages

- **Faster convergence** — accelerates in the consistent gradient direction
- **Reduces oscillations** — smooths out noisy gradients (especially in SGD)
- **Handles ravines well** — long narrow valleys where SGD bounces and Momentum glides through
- **Simple to implement** — just one extra hyperparameter (β)

## Disadvantages

- **Overshooting** — momentum can carry the update past the minimum, causing oscillations near it
- **Extra hyperparameter β** — usually 0.9 works, but needs tuning sometimes
- **Uniform learning rate** — still uses the same α for all parameters (no per-parameter adaptation)
- **Not adaptive** — doesn't adjust step size based on gradient history magnitude

---

## When to Use

- When vanilla SGD is converging too slowly
- When the loss landscape has narrow ravines or a lot of noise
- Good default: **β = 0.9**

---

## Key Takeaway

Momentum keeps a "memory" of which direction we've been going and continues in that direction, making optimization smoother and faster. It is almost always better than plain SGD.
