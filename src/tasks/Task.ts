export interface StepResult {
    nextState: number[];
    reward: number;
    done: boolean;
}

export interface Task {
    reset(): number[];
    step(actionValue: number): StepResult;
}