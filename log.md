## The training loop for double pendulum cart balancing with Q-table, and what actually went wrong 

The double pendulum plateaued early at a low score, and instead of just shrugging
and tweaking the learning rate, we chased it down properly. Roughly in order of
what we found:

1. **The Q-table was basically empty.** Out of ~14,000 possible discretized states,
   the agent had only ever visited 16 of them, with one single state-action pair
   accounting for over half of all training steps. The `resetWorld()` function
   started every episode from the exact same coordinates, so with a fully
   deterministic engine, the agent never saw a different starting point — it was
   stuck retracing the same tiny loop for millions of steps.
   - **Fix:** randomize the initial pole angles slightly on every reset.

2. **Coverage improved, but visits were still wildly lopsided**, and each state
   only had 8–11 samples of one of the two actions before the agent "decided" that
   action was bad and stopped trying it. That's not a real conclusion, that's
   noise from a handful of unlucky early rolls, and UCB's exploration bonus
   (`sqrt(log(totalSteps) / visits)`) grows too slowly to ever recover from a bad
   early estimate once the Q-value gap is large.
   - **Fix:** a hard exploration floor — force every action to be tried some
     minimum number of times per state before trusting Q-value comparisons.

3. **Reward was flat** (`+1` per surviving step, `-100` on death), which only gives
   the agent a signal at the moment of failure. It has no way to tell "barely
   hanging on" apart from "perfectly vertical" until it's too late.
   - **Fix:** shaped reward using `cos(angle)` for both poles, so the agent is
     continuously graded on how close to upright it is, not just whether it's
     alive.

4. **Symmetry.** A double pendulum cart-pole is left-right symmetric, so every
   experienced transition can be mirrored (flip angle/velocity bins, flip the
   action) and learned from twice. This doubled effective sample efficiency for
   free and fixed a case where the agent had wildly different confidence about
   mirror-image states that should have been identical.

5. **Finer bins helped, but showed diminishing returns fast.** Going from ~14,000
   to ~65,000 possible states (finer angle and velocity resolution) needed roughly
   12x more training to get less than 3x more state coverage. That ratio getting
   worse as the grid gets finer is the actual ceiling of tabular Q-learning on this
   problem — the states that most need precision (near the failure boundary) are
   the ones a decent policy visits least, so refining the grid multiplies the total
   space without proportionally filling it in.

A small diagnostic API was added to `QLearningAgent` along the way
(`getCoverageStats`, `getTopStates`) to actually measure this instead of guessing —
state visitation coverage, visit-count histograms, and the Q-values/visit counts
for the most-visited states. Worth keeping around for whatever comes next.


## Where this is going

Tabular Q-learning has hit its ceiling here — the exponential blowup of
`bins^dimensions`, combined with under-visited boundary states, is a structural
limit, not a tuning problem. 


Next step is **DQN**: replace the table with a network that takes the continuous state directly (no discretization), add a
replay buffer and target network for stability, and carry over the reward shaping
and symmetry-augmentation ideas from the tabular version, which should still
apply.
