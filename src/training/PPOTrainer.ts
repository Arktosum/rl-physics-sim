// src/training/PPOTrainer.ts

import { PPOAgent } from '../agents/PPOAgent';
import { PPORolloutBuffer } from './PPORolloutBuffer';
import type { Task } from '../tasks/Task';

const SCORE_WINDOW = 100;
// Dropped from 200 -> 30 (~0.5s of simulated time at fixed dt=0.016). At 200,
// the window spanned ~3.2s — long enough for the pendulum's state (and so the
// Actor's per-step `mean`) to drift a lot, making the action histogram show
// the spread of means across many different states rather than sampling
// noise around one state. That made it look inconsistent with the overlaid
// Gaussian(mean, std) curve, which only reflects the SINGLE most recent
// state. A shorter window keeps the state (and mean) closer to constant
// across it, so the histogram is now a fairer visual match for the curve.
const ACTION_HISTORY_LIMIT = 30;

const HORIZON = 2048;             // steps collected before pausing to train
// Raised from 2000 -> 4000 (~64s) to give recovery arcs room to actually play
// out now that DoublePendulumTask's fall condition is much more forgiving
// (tip-below-cart instead of a tight fixed angle). Deliberately left larger
// than HORIZON: an episode that outlives one training pause just continues
// seamlessly afterward (doOneStep()'s stepsThisEpisode counter doesn't reset
// on a training pause, only on a real episode end) — the GAE bootstrap value
// already handles a buffer that gets cut off mid-episode correctly, so
// HORIZON doesn't need to scale up just to "keep pace" with this.
const MAX_EPISODE_STEPS = 4000;   // Truncate episodes that go on too long

function pushCapped(arr: number[], value: number, limit: number): void {
    arr.push(value);
    if (arr.length > limit) arr.shift();
}

export class PPOTrainer {
    public episode = 1;
    public score = 0;
    public currentState: number[];

    // UI Diagnostics
    public currentAction = 0;
    // The Actor's raw distribution parameters for the state it just acted on
    // (BEFORE exploration noise/clamping) — mean is tanh-squashed to [-1, 1],
    // std is how far it's currently willing to sample away from that mean.
    // Exposed so the "action distribution" chart can be checked against the
    // actual numbers behind it instead of just the sampled-action histogram.
    public currentActionMean = 0;
    public currentActionStd = 0;
    public readonly actionHistory: number[] = [];
    public readonly scoreHistory: number[] = [];
    public readonly survivalTimeHistory: number[] = [];
    public currentMovingAvg = 0;
    public maxSurvivalTime = 0;

    // PPO Health Metrics
    public currentActorLoss = 0;
    public currentCriticLoss = 0;
    public currentAdvantage = 0;
    public currentClipFraction = 0;
    public currentKlDivergence = 0;

    // Perf diagnostics — how long the actual training math takes per HORIZON,
    // and how many physics steps we're managing to push per second overall.
    // Exposed so we can watch these live instead of guessing where time goes.
    public totalSteps = 0;
    public lastTrainMs = 0;
    public avgTrainMs = 0;

    public timeBudgetMs: number;
    public stepsThisEpisode = 0;
    private stepsSinceLastTrain = 0; // Tracks the Horizon
    // Set by doOneStep() when the Horizon is hit; consumed by tick() BETWEEN
    // physics steps rather than mid-loop, so the (now async, chunked) train()
    // call is always awaited from a clean point instead of firing off inside
    // a synchronous while loop.
    private needsTraining = false;

    private readonly agent: PPOAgent;
    private readonly task: Task;
    private readonly buffer: PPORolloutBuffer;
    // The environment's actual next-state, captured BEFORE any episode-end reset
    // overwrites currentState. Needed so train() can bootstrap GAE correctly when
    // the HORIZON cuts a rollout off mid-episode rather than at a real termination.
    private lastNextState: number[];

    constructor(agent: PPOAgent, task: Task, timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.timeBudgetMs = timeBudgetMs;
        this.buffer = new PPORolloutBuffer();
        this.currentState = task.reset();
        this.lastNextState = this.currentState;
    }

    private doOneStep(): void {
        const { rawAction, clampedAction, logProb, value, mean, std } = this.agent.act(this.currentState);
        this.currentAction = clampedAction;
        this.currentActionMean = mean;
        this.currentActionStd = std;

        pushCapped(this.actionHistory, clampedAction, ACTION_HISTORY_LIMIT);

        const { nextState, reward, done } = this.task.step(clampedAction);
        this.lastNextState = nextState; // captured before the done-branch below can call task.reset()

        // store rawAction (pre-clamp) since PPOAgent.learn's gradient math needs it
        this.buffer.add({
            state: this.currentState,
            action: rawAction,
            reward,
            value,
            logProb,
            done
        });

        this.stepsThisEpisode++;
        this.totalSteps++;
        this.score += reward;

        // episode end just resets state here; training only happens at the horizon check below
        if (done || this.stepsThisEpisode >= MAX_EPISODE_STEPS) {
            pushCapped(this.scoreHistory, this.score, SCORE_WINDOW);
            this.currentMovingAvg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;

            const survivalSeconds = this.stepsThisEpisode * 0.016; // fixed dt
            if (survivalSeconds > this.maxSurvivalTime) this.maxSurvivalTime = survivalSeconds;
            pushCapped(this.survivalTimeHistory, survivalSeconds, SCORE_WINDOW);

            this.score = 0;
            this.stepsThisEpisode = 0;
            this.episode++;
            this.currentState = this.task.reset();
        } else {
            this.currentState = nextState;
        }

        // flag training here; tick() actually calls train() once its physics
        // loop below has stopped, not from inside it
        this.stepsSinceLastTrain++;
        if (this.stepsSinceLastTrain >= HORIZON) {
            this.stepsSinceLastTrain = 0;
            this.needsTraining = true;
        }
    }

    private async train(): Promise<void> {
        const trainStart = performance.now();

        // critic's value estimate for the state after the last stored transition;
        // safe to compute even if that transition was terminal since PPOAgent.learn's
        // nextNonTerminal gate zeroes it out in that case
        const bootstrapValue = this.agent.getValue(this.lastNextState);

        // learn() yields internally every few hundred samples so this doesn't
        // block the thread's own timers for its whole ~6000-sample duration
        const metrics = await this.agent.learn(this.buffer, bootstrapValue);

        if (!Number.isNaN(metrics.actorLoss)) {
            this.currentActorLoss = metrics.actorLoss;
            this.currentCriticLoss = metrics.criticLoss;
            this.currentAdvantage = metrics.avgAdvantage;
            this.currentClipFraction = metrics.clipFraction;
            this.currentKlDivergence = metrics.klDivergence;
        }

        // PPO is on-policy: never reuse transitions across an update
        this.buffer.clear();

        // EMA so a single slow call (e.g. first JIT-cold call) doesn't dominate the display
        this.lastTrainMs = performance.now() - trainStart;
        const alpha = 0.2;
        this.avgTrainMs = this.avgTrainMs === 0
            ? this.lastTrainMs
            : alpha * this.lastTrainMs + (1 - alpha) * this.avgTrainMs;
    }

    // Does ONE burst of work (however many steps fit in the time budget, plus
    // a training pause if the horizon was hit) and returns — does NOT
    // self-reschedule. The caller (the Worker's runTrainingWorker harness)
    // owns the "keep going forever" loop, so this can be driven by a test
    // harness or any other caller without an implicit setTimeout tied to it.
    public tick = async (): Promise<void> => {
        const start = performance.now();
        while (performance.now() - start < this.timeBudgetMs && !this.needsTraining) {
            this.doOneStep();
        }

        if (this.needsTraining) {
            this.needsTraining = false;
            await this.train();
        }
    };
}

/**
 * The subset of PPOTrainer's public fields PPODOMUI actually reads.
 * PPOTrainer satisfies this structurally already (no code change needed there).
 * This exists so a plain-data mirror object on the main thread — fed by
 * postMessage from a training Worker — can be handed to PPODOMUI too,
 * without PPODOMUI needing to know whether it's looking at a live trainer
 * or a snapshot relayed across a thread boundary.
 */
export interface PPOTrainerLike {
    episode: number;
    score: number;
    stepsThisEpisode: number;
    maxSurvivalTime: number;
    currentCriticLoss: number;
    currentAdvantage: number;
    currentClipFraction: number;
    currentKlDivergence: number;
    actionHistory: number[];
    scoreHistory: number[];
    survivalTimeHistory: number[];

    // The Actor's live distribution parameters, plus the actual physical
    // force those numbers translate to. currentActionMean/currentActionStd
    // come straight from the trainer; currentThrustNewtons/maxThrustNewtons
    // are computed by the Worker (only it has the concrete Task instance and
    // therefore knows the Actuator's real thrustPower — PPOTrainer stays
    // task-agnostic and never sees that value itself).
    currentActionMean: number;
    currentActionStd: number;
    currentThrustNewtons: number;
    maxThrustNewtons: number;

    // Perf diagnostics. totalSteps/lastTrainMs/avgTrainMs come straight from
    // the trainer; stepsPerSecond and maxWorkerFrameGapMs are computed by the
    // Worker (it's the only place with a clock on both sides of the gap);
    // mainThreadFrameGapMs is written directly by the main thread's own
    // render loop, never touched by the 'metrics' postMessage handler.
    totalSteps: number;
    lastTrainMs: number;
    avgTrainMs: number;
    stepsPerSecond: number;
    maxWorkerFrameGapMs: number;
    mainThreadFrameGapMs: number;

    // Live Eval: a second physics environment, run entirely separately from
    // training, driven by the CURRENT policy's greedy (noise-free) action —
    // "how long does the agent survive when actually trying its best," as
    // opposed to scoreHistory/maxSurvivalTime above which include training's
    // deliberate exploration noise.
    evalSurvivalSeconds: number;
    evalMaxSurvivalSeconds: number;
}