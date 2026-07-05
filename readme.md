# physics-sim

A physics engine built from scratch in TypeScript, used as a testbed for building physics environments and teaching reinforcement learning agents on. No physics libraries, no RL libraries — the point of this project wasto actually understand both halves rather than import them.

## What's in here

### The physics engine

- **`Vector.ts`** — basic 2D vector math.
- **`PointMass.ts`** — a point mass integrated with **Verlet integration**. Instead
  of tracking velocity directly, it keeps the previous position and derives velocity
  implicitly from the difference between old and current position. This makes it
  trivial to plug into a constraint solver without velocity bookkeeping getting out
  of sync.
- **`Constraint.ts`** — distance constraints (rods) solved with **Position-Based
  Dynamics**: each constraint nudges connected points back to the correct distance,
  weighted by inverse mass, so pinned/heavy points don't get shoved around by light
  ones. Also includes axis constraints (for the cart, which only moves horizontally)
  and boundary constraints (track edges).
- **`Environment.ts`** — the simulation loop: apply gravity, integrate positions,
  then relax all constraints for a few iterations per frame. Standard PBD structure.
- **`Actuator.ts`** — applies horizontal force to the cart based on the agent's
  chosen action.
- **`CanvasRenderer.ts`** — draws all of it to a `<canvas>`.

This part works and is reasonably solid. Verlet + PBD is the same family of
technique used in cloth and rope sims, and getting the mass-weighting right on the
constraints was the main thing worth being careful about.

### The RL agent

- **`QLearningAgent.ts`** — tabular Q-learning, epsilon-greedy → later swapped to
  UCB-style exploration. States are discretized (cart position, both pole angles,
  both angular velocities) and stored in a `Map<string, number[]>`.
- **`main.ts`** — the double-pendulum cart-pole training loop.
- **`singePendulum_main.ts`** — the single-pendulum version, an earlier, simpler
  version of the same idea.


## Running it

```bash
npm install
npm run dev
```
