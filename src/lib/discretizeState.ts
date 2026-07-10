// Turns a Task's continuous, already-normalized state array into a discrete
// string key a tabular Q-learning agent can use as a table index.
// Task-agnostic — works for SinglePendulumTask's 4-dim state and
// DoublePendulumTask's 8-dim state alike, since every Task normalizes each
// dimension to roughly [-1, 1] already.

export function binVariable(value: number, min: number, max: number, bins: number): number {
    const clamped = Math.max(min, Math.min(max, value));
    const normalized = (clamped - min) / (max - min);
    const bin = Math.floor(normalized * bins);
    return Math.min(bin, bins - 1);
}

/**
 * binsPerDim controls resolution per state dimension — fewer dimensions can
 * afford more bins each and stay expressive; more dimensions need fewer
 * bins each to keep the total state space (the product of every entry)
 * tractable for a table. See docs/journey/01-q-learning.md for why that
 * product growing faster than coverage can fill it in is the actual ceiling
 * of this approach.
 */
export function discretizeState(state: number[], binsPerDim: number[]): string {
    if (state.length !== binsPerDim.length) {
        throw new Error(`discretizeState: state has ${state.length} dims, binsPerDim has ${binsPerDim.length}`);
    }
    const parts = new Array(state.length);
    for (let i = 0; i < state.length; i++) {
        parts[i] = binVariable(state[i], -1, 1, binsPerDim[i]);
    }
    return parts.join('-');
}

export function totalStateSpaceSize(binsPerDim: number[]): number {
    return binsPerDim.reduce((a, b) => a * b, 1);
}
