import { QLearningAgent } from '../agents/QLearningAgent';
import type { Task } from '../tasks/Task';
import { discretizeState, totalStateSpaceSize } from '../lib/discretizeState';

const SCORE_WINDOW = 100;
const ACTION_VALUES = [-1, 1]; // discrete action index -> thrust fraction sent to the Task

function pushCapped(arr: number[], value: number, limit: number): void {
    arr.push(value);
    if (arr.length > limit) arr.shift();
}

export interface QLearningTaskConfig {
    /** Bins per state dimension — see discretizeState.ts for the sizing trade-off. */
    binsPerDim: number[];
    /**
     * Which state dimensions flip sign under a left-right mirror. Needed for
     * the symmetry-augmentation trick (docs/journey/01-q-learning.md #4):
     * every transition is left-right symmetric, so it can be learned from
     * twice — once as recorded, once mirrored — for free extra sample
     * efficiency. Position/velocity dimensions and sin(angle) terms flip
     * (odd functions of the mirror); cos(angle) terms don't (even functions).
     */
    mirrorMask: boolean[];
}

/**
 * Drives the act/step/learn cycle for tabular Q-learning, reusing the same
 * Task abstraction (reset()/step()) every other algorithm in this project
 * uses — the physics and reward shaping are NOT reimplemented here the way
 * the old (deleted) doublePendulum_main.ts duplicated them inline. State
 * discretization and the mirror-symmetry trick are what's specific to this
 * trainer; everything else is standard.
 */
export class QLearningTrainer {
    public episode = 1;
    public score = 0;
    public currentState: number[];
    public totalSteps = 0;
    public stepsThisEpisode = 0;
    public maxSurvivalTime = 0;
    public readonly scoreHistory: number[] = [];
    public readonly survivalTimeHistory: number[] = [];
    public currentMovingAvg = 0;

    // Coverage stats — refreshed periodically (§refreshCoverageStats), not
    // every step: getCoverageStats() walks the whole visit-count table, and
    // there's no reason to pay that cost at 10Hz.
    public statesVisited = 0;
    public coverageFraction = 0;
    public singleVisitFraction = 0;
    public totalVisits = 0;

    private readonly agent: QLearningAgent;
    private readonly task: Task;
    private readonly binsPerDim: number[];
    private readonly mirrorMask: boolean[];
    private readonly totalPossibleStates: number;
    public timeBudgetMs: number;
    private readonly dt = 0.016;

    private currentStateKey: string;

    constructor(agent: QLearningAgent, task: Task, config: QLearningTaskConfig, timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.binsPerDim = config.binsPerDim;
        this.mirrorMask = config.mirrorMask;
        this.totalPossibleStates = totalStateSpaceSize(this.binsPerDim);
        this.timeBudgetMs = timeBudgetMs;
        this.currentState = task.reset();
        this.currentStateKey = discretizeState(this.currentState, this.binsPerDim);
    }

    private mirror(state: number[]): number[] {
        return state.map((v, i) => this.mirrorMask[i] ? -v : v);
    }

    private doOneStep(): void {
        const action = this.agent.getAction(this.currentStateKey);
        const { nextState, reward, done } = this.task.step(ACTION_VALUES[action]);
        const nextStateKey = discretizeState(nextState, this.binsPerDim);

        this.agent.learn(this.currentStateKey, action, reward, nextStateKey, done);

        // Symmetry augmentation — learn the mirrored transition too.
        const mirroredCurrentKey = discretizeState(this.mirror(this.currentState), this.binsPerDim);
        const mirroredNextKey = discretizeState(this.mirror(nextState), this.binsPerDim);
        this.agent.learn(mirroredCurrentKey, 1 - action, reward, mirroredNextKey, done);

        this.totalSteps++;
        this.stepsThisEpisode++;
        this.score += reward;

        if (done) {
            this.onEpisodeEnd();
        } else {
            this.currentState = nextState;
            this.currentStateKey = nextStateKey;
        }
    }

    private onEpisodeEnd(): void {
        const survivalSeconds = this.stepsThisEpisode * this.dt;
        if (survivalSeconds > this.maxSurvivalTime) this.maxSurvivalTime = survivalSeconds;
        pushCapped(this.survivalTimeHistory, survivalSeconds, SCORE_WINDOW);
        this.stepsThisEpisode = 0;

        pushCapped(this.scoreHistory, this.score, SCORE_WINDOW);
        this.currentMovingAvg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
        this.score = 0;
        this.episode++;

        this.currentState = this.task.reset();
        this.currentStateKey = discretizeState(this.currentState, this.binsPerDim);

        if (this.episode % 20 === 0) this.refreshCoverageStats();
    }

    private refreshCoverageStats(): void {
        const stats = this.agent.getCoverageStats(this.totalPossibleStates);
        this.statesVisited = stats.statesVisited;
        this.coverageFraction = stats.coverageFraction ?? 0;
        this.singleVisitFraction = stats.singleVisitFraction;
        this.totalVisits = stats.totalVisits;
    }

    // Does one burst of work and returns — the Worker harness owns the
    // reschedule loop, same contract as the other trainers.
    public tick = (): void => {
        const start = performance.now();
        while (performance.now() - start < this.timeBudgetMs) {
            this.doOneStep();
        }
    };
}
