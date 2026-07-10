import { DQNAgent } from '../agents/DQNAgent';
import type { Task } from '../tasks/Task';

const SCORE_WINDOW = 100;
const CHART_HISTORY_LIMIT = 200;
const METRIC_SAMPLE_RATE = 0.05;
const EMA_WEIGHT = 0.01;

function pushCapped(arr: number[], value: number, limit: number): void {
    arr.push(value);
    if (arr.length > limit) arr.shift();
}

/**
 * Same job as the original (main-thread) Trainer.ts — drive the act/step/
 * remember/replay cycle, expose diagnostics fields — adapted to the Worker
 * harness contract: tick() does one burst and returns rather than
 * self-rescheduling, and totalSteps is exposed for the harness's
 * stepsPerSecond computation.
 */
export class DQNTrainer {
    public episode = 1;
    public score = 0;
    public maxScore = 0;
    public currentActionIndex = 0;
    public currentState: number[];

    public readonly scoreHistory: number[] = [];
    public readonly lossHistory: number[] = [];
    public readonly qValueHistory: number[] = [];
    public readonly movingAverageHistory: number[] = [];

    public currentLoss = 0;
    public currentQ = 0;
    public currentMovingAvg = 0;
    public maxMovingAvg = 0;

    public totalSteps = 0;

    private readonly agent: DQNAgent;
    private readonly task: Task;
    private readonly thrustLevels: number[];
    public timeBudgetMs: number;

    public stepsThisEpisode = 0;
    public maxSurvivalTime = 0;
    public currentAvgSurvivalTime = 0;
    public readonly survivalTimeHistory: number[] = [];
    private readonly dt = 0.016;

    constructor(agent: DQNAgent, task: Task, thrustLevels: number[], timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.thrustLevels = thrustLevels;
        this.timeBudgetMs = timeBudgetMs;
        this.currentState = task.reset();
    }

    private doOneStep(): void {
        this.currentActionIndex = this.agent.getAction(this.currentState);
        const thrustFraction = this.thrustLevels[this.currentActionIndex];

        const { nextState, reward, done } = this.task.step(thrustFraction);
        this.agent.remember(this.currentState, this.currentActionIndex, reward, nextState, done);

        this.totalSteps++;
        this.stepsThisEpisode++;

        // train every 4 steps rather than every step (compute budget)
        let metrics = null;
        if (this.totalSteps % 4 === 0) {
            metrics = this.agent.replay();
        }
        if (metrics && !Number.isNaN(metrics.loss)) {
            this.currentLoss = this.currentLoss * (1 - EMA_WEIGHT) + metrics.loss * EMA_WEIGHT;
            this.currentQ = this.currentQ * (1 - EMA_WEIGHT) + metrics.qValue * EMA_WEIGHT;
            if (Math.random() < METRIC_SAMPLE_RATE) {
                pushCapped(this.lossHistory, this.currentLoss, CHART_HISTORY_LIMIT);
                pushCapped(this.qValueHistory, this.currentQ, CHART_HISTORY_LIMIT);
            }
        }

        if (done) {
            this.onEpisodeEnd();
        } else {
            this.score += reward;
            this.currentState = nextState;
        }
    }

    private onEpisodeEnd(): void {
        const survivalSeconds = this.stepsThisEpisode * this.dt;
        if (survivalSeconds > this.maxSurvivalTime) this.maxSurvivalTime = survivalSeconds;
        pushCapped(this.survivalTimeHistory, survivalSeconds, 100);
        this.currentAvgSurvivalTime = this.survivalTimeHistory.reduce((a, b) => a + b, 0) / this.survivalTimeHistory.length;
        this.stepsThisEpisode = 0;

        if (this.score > this.maxScore) this.maxScore = this.score;
        pushCapped(this.scoreHistory, this.score, SCORE_WINDOW);

        this.currentMovingAvg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
        if (this.currentMovingAvg > this.maxMovingAvg) this.maxMovingAvg = this.currentMovingAvg;
        pushCapped(this.movingAverageHistory, this.currentMovingAvg, CHART_HISTORY_LIMIT);

        this.agent.decayEpsilon();
        this.currentState = this.task.reset();
        this.episode++;
        this.score = 0;
    }

    // Does one burst of work and returns — the Worker harness owns the
    // reschedule loop, same contract as PPOTrainer/ReinforceTrainer.
    public tick = (): void => {
        const start = performance.now();
        while (performance.now() - start < this.timeBudgetMs) {
            this.doOneStep();
        }
    };
}
