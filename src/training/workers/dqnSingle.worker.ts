// src/training/workers/dqnSingle.worker.ts — same shape as dqnDouble.worker.ts, SinglePendulumTask instead.

import { DQNAgent } from '../../agents/DQNAgent';
import { DQNTrainer } from '../DQNTrainer';
import { SinglePendulumTask } from '../../tasks/SinglePendulumTask';
import { runTrainingWorker } from './workerHarness';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const TRACK_HEIGHT = CANVAS_HEIGHT - 150;
const FIXED_DT = 0.016;
const ENERGY_PENALTY_WEIGHT = 0.02;
const THRUST_LEVELS = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0];

const task = new SinglePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT, ENERGY_PENALTY_WEIGHT);
const initialState = task.reset();
const agent = new DQNAgent(initialState.length, THRUST_LEVELS.length);

const TRAIN_TIME_BUDGET_MS = 20;
const trainer = new DQNTrainer(agent, task, THRUST_LEVELS, TRAIN_TIME_BUDGET_MS);

const evalTask = new SinglePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT, ENERGY_PENALTY_WEIGHT);
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
        currentLoss: trainer.currentLoss,
        currentQ: trainer.currentQ,
        lossHistory: trainer.lossHistory,
        qValueHistory: trainer.qValueHistory,
        latestQValues: agent.getQValues(trainer.currentState),
        currentActionIndex: trainer.currentActionIndex,
        epsilon: agent.epsilon,
        scoreHistory: trainer.scoreHistory,
        survivalTimeHistory: trainer.survivalTimeHistory,
        lastTrainMs: 0,
        avgTrainMs: 0,
        evalSurvivalSeconds: evalStepsThisEpisode * FIXED_DT,
        evalMaxSurvivalSeconds,
    }),
    runEvalStep: () => {
        const actionIndex = agent.getGreedyAction(evalState);
        const { nextState, done } = evalTask.step(THRUST_LEVELS[actionIndex]);
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
