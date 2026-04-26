# Nesterov Accelerated Gradient (NAG)

**Date:** 2026-04-26

---

## What is NAG?

**Nesterov Accelerated Gradient** is an improvement over Momentum SGD. The key difference: instead of computing the gradient at the **current position**, NAG computes it at the **lookahead position** — where the momentum would take you next.

It's like Momentum with a "look before you leap" strategy.

---

## Intuition

- **Momentum SGD**: "Move in the direction I've been going, then correct based on the gradient here."
- **NAG**: "First, take the momentum step to see where I'll end up, then correct from **that** position."

By computing the gradient at the future position, NAG anticipates and brakes early when approaching a minimum, reducing overshooting.

---

## Mathematics

```
θ_lookahead = θ - β * v          ← peek ahead using current velocity
v  = β * v  +  α * ∂L/∂θ_lookahead   ← gradient at lookahead position
θ  = θ - v
```

Or equivalently in the standard form:

```
v_new = β * v  +  α * ∇L(θ - β * v)
θ     = θ - v_new
```

- **β** — momentum coefficient (typically 0.9)
- **α** — learning rate
- **∇L(θ - β * v)** — gradient at the lookahead point, not the current point

---

## Advantages

- **More accurate updates** — gradient computed at a better (future) position
- **Less overshooting** — anticipates the minimum and corrects earlier than Momentum
- **Faster convergence than Momentum SGD** — especially in convex problems
- **Theoretically optimal** for convex functions (provably better convergence rate)

## Disadvantages

- **Slightly more complex** to implement than standard Momentum
- **Same uniform learning rate** issue — no per-parameter adaptation
- **Marginal improvement** over Momentum in practice for deep networks (most benefits seen in convex settings)
- **Not adaptive** — still doesn't scale learning rate by gradient magnitude

---

## Momentum SGD vs NAG — Key Difference

| | Momentum SGD | NAG |
|---|---|---|
| Where gradient is computed | Current position θ | Lookahead position θ - β·v |
| Anticipates minimum? | No | Yes |
| Overshooting | More | Less |

---

## When to Use

- When Momentum SGD overshoots or oscillates near the minimum
- Convex optimization problems (theoretical guarantees apply)
- **β = 0.9** is standard

---

## Key Takeaway

NAG is a smarter version of Momentum — it peeks ahead, computes the gradient from the future position, and uses that to correct course. This small change makes convergence faster and reduces overshooting.
