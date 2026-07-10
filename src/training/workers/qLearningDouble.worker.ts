// src/training/workers/qLearningDouble.worker.ts — tabular Q-learning on the double pendulum.

import { QLearningAgent } from '../../agents/QLearningAgent';
import { QLearningTrainer } from '../QLearningTrainer';
import { DoublePendulumTask } from '../../tasks/DoublePendulumTask';
import { discretizeState } from '../../lib/discretizeState';
import { runTrainingWorker } from './workerHarness';

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const TRACK_HEIGHT = CANVAS_HEIGHT - 150;
const FIXED_DT = 0.016;
const ACTION_VALUES = [-1, 1];

// State: [cartX, cartV, sinA1, cosA1, sinA2, cosA2, v1, v2]. Bins chosen to
// land the total state space around ~20k (see docs/journey/01-q-learning.md
// for why that's roughly the ceiling this approach can meaningfully cover).
const BINS_PER_DIM = [3, 3, 4, 4, 4, 4, 3, 3];
const MIRROR_MASK = [true, true, true, false, true, false, true, true];

const task = new DoublePendulumTask(CANVAS_WIDTH, CANVAS_HEIGHT, TRACK_HEIGHT, FIXED_DT);
task.reset();
const agent = new QLearningAgent(ACTION_VALUES.length);

const TRAIN_TIME_BUDGET_MS = 20;
const trainer = new QLearningTrainer(agent, task, { binsPerDim: BINS_PER_DIM, mirrorMask: MIRROR_MASK }, TRAIN_TIME_BUDGET_MS);

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
        statesVisited: trainer.statesVisited,
        coverageFraction: trainer.coverageFraction,
        singleVisitFraction: trainer.singleVisitFraction,
        totalVisits: trainer.totalVisits,
        scoreHistory: trainer.scoreHistory,
        survivalTimeHistory: trainer.survivalTimeHistory,
        lastTrainMs: 0, // Q-learning updates the table inline every step — no separate chunked phase to time
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
