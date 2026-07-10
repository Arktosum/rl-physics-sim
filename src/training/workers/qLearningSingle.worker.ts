// src/training/workers/qLearningSingle.worker.ts — tabular Q-learning on the single pendulum.

import { QLearningAgent } from '../../agents/QLearningAgent';
import { QLearningTrainer } from '../QLearningTrainer';
import { SinglePendulumTask } from '../../tasks/SinglePendulumTask';
import { discretizeState } from '../../lib/discretizeState';
import { runTrainingWorker } from './workerHarness';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const TRACK_HEIGHT = CANVAS_HEIGHT - 150;
const FIXED_DT = 0.016;
const ENERGY_PENALTY_WEIGHT = 0.02;
const ACTION_VALUES = [-1, 1];

// State: [cartX, cartV, angle, angularVel] — no sin/cos pair here (unlike the
// double pendulum), so every dimension flips sign under a mirror. Much
// smaller state space than the double pendulum can afford finer bins.
const BINS_PER_DIM = [6, 6, 12, 10];
const MIRROR_MASK = [true, true, true, true];

const task = new SinglePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT, ENERGY_PENALTY_WEIGHT);
task.reset();
const agent = new QLearningAgent(ACTION_VALUES.length);

const TRAIN_TIME_BUDGET_MS = 20;
const trainer = new QLearningTrainer(agent, task, { binsPerDim: BINS_PER_DIM, mirrorMask: MIRROR_MASK }, TRAIN_TIME_BUDGET_MS);

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
        statesVisited: trainer.statesVisited,
        coverageFraction: trainer.coverageFraction,
        singleVisitFraction: trainer.singleVisitFraction,
        totalVisits: trainer.totalVisits,
        scoreHistory: trainer.scoreHistory,
        survivalTimeHistory: trainer.survivalTimeHistory,
        lastTrainMs: 0,
        avgTrainMs: 0,
        evalSurvivalSeconds: evalStepsThisEpisode * FIXED_DT,
        evalMaxSurvivalSeconds,
    }),
    runEvalStep: () => {
        const stateKey = discretizeState(evalState, BINS_PER_DIM);
        const actionIndex = agent.getGreedyAction(stateKey);
        const { nextState, done } = evalTask.step(ACTION_VALUES[actionIndex]);
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
