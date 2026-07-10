// src/training/PPORolloutBuffer.ts

export interface PPOTransition {
    state: number[];
    action: number;   // raw, unclamped action
    reward: number;
    value: number;     // critic's baseline prediction at this step
    logProb: number;   // actor's log-prob of this action, under the policy at collection time
    done: boolean;
}

export class PPORolloutBuffer {
    public transitions: PPOTransition[] = [];

    public add(transition: PPOTransition) {
        this.transitions.push(transition);
    }

    public get(): PPOTransition[] {
        return this.transitions;
    }

    public clear() {
        this.transitions = [];
    }

    public get length(): number {
        return this.transitions.length;
    }
}