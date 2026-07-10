// src/training/workers/reinforceDouble.worker.ts — REINFORCE on the double pendulum.

import { ReinforceAgent } from '../../agents/ReinforceAgent';
import { ReinforceTrainer } from '../ReinforceTrainer';
import { DoublePendulumTask } from '../../tasks/DoublePendulumTask';
import { runTrainingWorker } from './workerHarness';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const TRACK_HEIGHT = CANVAS_HEIGHT - 150;
const FIXED_DT = 0.016;

const task = new DoublePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT);
const initialState = task.reset();
const agent = new ReinforceAgent(initialState.length);

const TRAIN_TIME_BUDGET_MS = 20;
const trainer = new ReinforceTrainer(agent, task, TRAIN_TIME_BUDGET_MS);

const evalTask = new DoublePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT);
let evalState = evalTask.reset();
let evalStepsThisEpisode = 0;
let evalMaxSurvivalSeconds = 0;

function buildFrame(previewMode: 'training' | 'live') {
    const env = (previewMode === 'live' ? evalTask : task).env;
    return {
        points: env.points.map(p => ({
            position: { x: p.position.x, y: p.position.y },
            mass: p.mass,
            isPinned: p.isPinned,
        })),
        constraints: env.constraints
            .map((c: any) => {
                if ('p1' in c && 'p2' in c) {
                    return {
                        p1: { position: { x: c.p1.position.x, y: c.p1.position.y } },
                        p2: { position: { x: c.p2.position.x, y: c.p2.position.y } },
                    };
                }
                if ('lockedY' in c) return { lockedY: c.lockedY };
                return null;
            })
            .filter((c: unknown) => c !== null),
    };
}

runTrainingWorker({
    tick: () => trainer.tick(),
    buildFrame,
    buildMetrics: () => ({
        episode: trainer.episode,
        score: trainer.score,
        stepsThisEpisode: trainer.stepsThisEpisode,
        maxSurvivalTime: trainer.maxSurvivalTime,
        currentCriticLoss: trainer.currentCriticLoss,
        currentAdvantage: trainer.currentAdvantage,
        currentEntropy: trainer.currentEntropy,
        currentGradientClipRate: trainer.currentGradientClipRate,
        actionHistory: trainer.actionHistory,
        scoreHistory: trainer.scoreHistory,
        survivalTimeHistory: trainer.survivalTimeHistory,
        currentMean: trainer.currentMean,
        currentStd: trainer.currentStd,
        lastTrainMs: 0, // REINFORCE trains once per episode, synchronously, inline — no separate chunked phase to time
        avgTrainMs: 0,
        evalSurvivalSeconds: evalStepsThisEpisode * FIXED_DT,
        evalMaxSurvivalSeconds,
    }),
    runEvalStep: () => {
        const { clampedAction } = agent.actGreedy(evalState);
        const { nextState, done } = evalTask.step(clampedAction);
        evalStepsThisEpisode++;
        if (done) {
            const survivalSeconds = evalStepsThisEpisode * FIXED_DT;
            if (survivalSeconds > evalMaxSurvivalSeconds) evalMaxSurvivalSeconds = survivalSeconds;
            evalStepsThisEpisode = 0;
            evalState = evalTask.reset();
        } else {
            evalState = nextState;
        }
    },
    getTotalSteps: () => trainer.totalSteps,
    getEpisode: () => trainer.episode,
    toJSON: () => agent.toJSON(),
    loadJSON: json => agent.loadJSON(json),
});
