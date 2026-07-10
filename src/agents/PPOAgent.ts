import { DenseLayer } from "../lib/DenseLayer";
import { NeuralNetwork } from "../lib/NeuralNetwork";
import { ReLULayer } from "../lib/ReLULayer";
import { Matrix } from "../lib/Matrix";
import { clamp, sampleStandardNormal } from "../lib/mathUtils";
import type { PPORolloutBuffer } from "../training/PPORolloutBuffer";

const LOG_STD_MIN = -3.0;
const LOG_STD_MAX = 0.5;

// epochs * length single-sample updates (e.g. 3*2048=6144) was the worst
// shape for a JS engine: thousands of tiny function calls instead of a few
// wide matrix ops. 64 is also the standard PPO minibatch size (Stable-
// Baselines3 default); batching cuts weight-update calls ~64x
// (2048/64 * 3 = 96 batched updates instead of 6144 single-sample ones).
const MINIBATCH_SIZE = 64;

// yielding once per minibatch, not every N samples, keeps the worker's own
// timers (frame/metrics postMessage) responsive with little overhead since
// there are far fewer iterations to yield inside now
const yieldToEventLoop = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0));

export class PPOAgent {
    public actor: NeuralNetwork;
    public critic: NeuralNetwork;

    // LRs tuned for the old single-sample-SGD regime (6144 full-strength
    // updates per learn() call). Mini-batching averages the gradient across
    // each 64-sample batch, which silently made every update ~64x weaker
    // without a compensating LR bump. Verified on SinglePendulumTask: at the
    // un-bumped rates, KL divergence and clip fraction stayed pinned at ~0
    // for 1000+ episodes (actor wasn't learning). A 16x bump — conservative
    // vs. the full linear-scaling-rule 64x — restored healthy learning.
    public actorLearningRate: number = 0.008;   // 0.0005 * 16
    public criticLearningRate: number = 0.016;  // 0.001 * 16
    public gamma: number = 0.99;
    public lam: number = 0.95;                  // GAE lambda
    public clipRatio: number = 0.2;             // max 20% policy change per update

    private readonly inputSize: number;

    // batch-shaped scratch, owned by the agent since only learn() has the
    // context to assemble per-sample state arrays into a minibatch. sized
    // lazily, reused across minibatches unless width changes (last/ragged
    // minibatch of a learn() call)
    private stateBatchScratch: Matrix | null = null;
    private criticGradBatchScratch: Matrix | null = null;
    private actorGradBatchScratch: Matrix | null = null;

    constructor(inputSize: number) {
        this.inputSize = inputSize;
        // hidden width dropped 64 -> 32: the 64x64 middle layer was ~85% of
        // this network's FLOPs (4096 weights vs 512+128 for input/output
        // combined), the one layer where shrinking it actually cuts compute.
        // 32 is still generous for an 8-dim state; bump back up if
        // score/survival plateaus noticeably lower than before.
        const HIDDEN = 32;
        this.actor = new NeuralNetwork([
            new DenseLayer(inputSize, HIDDEN), new ReLULayer(),
            new DenseLayer(HIDDEN, HIDDEN), new ReLULayer(),
            new DenseLayer(HIDDEN, 2) // [meanRaw, logStdRaw]
        ]);

        this.critic = new NeuralNetwork([
            new DenseLayer(inputSize, HIDDEN), new ReLULayer(),
            new DenseLayer(HIDDEN, HIDDEN), new ReLULayer(),
            new DenseLayer(HIDDEN, 1) // [V(s)]
        ]);
    }

    private calculateLogProb(action: number, mean: number, std: number): number {
        const variance = std * std;
        const diff = action - mean;
        return -0.5 * ((diff * diff) / variance + Math.log(2 * Math.PI * variance));
    }

    private decodeActorOutput(state: number[]): { meanRaw: number; mean: number; logStdRaw: number; std: number } {
        const [meanRaw, logStdRawUnclamped] = this.actor.predict(state);
        const logStdRaw = clamp(logStdRawUnclamped, LOG_STD_MIN, LOG_STD_MAX);
        const mean = Math.tanh(meanRaw);
        const std = Math.exp(logStdRaw);
        return { meanRaw, mean, logStdRaw, std };
    }

    // queries actor and critic together so the rollout buffer can record
    // exactly what the brain was thinking at this step
    public act(state: number[]): {
        rawAction: number;
        clampedAction: number;
        logProb: number;
        value: number;
        mean: number;   // pre-noise distribution center, tanh-squashed to [-1, 1]
        std: number;    // exploration spread around mean
    } {
        const { mean, std } = this.decodeActorOutput(state);
        const z = sampleStandardNormal();
        const rawAction = mean + std * z;
        const clampedAction = clamp(rawAction, -1, 1);

        const logProb = this.calculateLogProb(rawAction, mean, std);
        const value = this.critic.predict(state)[0];

        return { rawAction, clampedAction, logProb, value, mean, std };
    }

    // skips the Gaussian sampling in act() — for demoing current policy
    // behavior rather than training-time exploration
    public actGreedy(state: number[]): { clampedAction: number; mean: number; std: number } {
        const { mean, std } = this.decodeActorOutput(state);
        return { clampedAction: clamp(mean, -1, 1), mean, std };
    }

    public getValue(state: number[]): number {
        return this.critic.predict(state)[0];
    }

    public async learn(buffer: PPORolloutBuffer, bootstrapValue: number = 0): Promise<{
        actorLoss: number;
        criticLoss: number;
        avgAdvantage: number;
        clipFraction: number;
        klDivergence: number
    }> {
        const episode = buffer.get();
        const length = episode.length;

        const advantages = new Array(length).fill(0);
        const returns = new Array(length).fill(0);

        let lastAdvantage = 0;
        // rollout is filled by a fixed horizon, not by waiting for the episode
        // to actually terminate, so most of the time it's truncated mid-episode.
        // the true future value there isn't 0, it's the critic's estimate for
        // the next state (bootstrapValue). if the last transition genuinely
        // was terminal, nextNonTerminal zeroes this out on first use below, so
        // passing bootstrapValue unconditionally is always safe.
        let lastValue = bootstrapValue;

        for (let t = length - 1; t >= 0; t--) {
            const transition = episode[t];
            const nextNonTerminal = transition.done ? 0 : 1;

            const delta = transition.reward + this.gamma * lastValue * nextNonTerminal - transition.value;
            lastAdvantage = delta + this.gamma * this.lam * nextNonTerminal * lastAdvantage;
            advantages[t] = lastAdvantage;

            returns[t] = advantages[t] + transition.value;
            lastValue = transition.value;
        }

        const advMean = advantages.reduce((a, b) => a + b, 0) / length;
        const advVariance = advantages.reduce((sum, a) => sum + Math.pow(a - advMean, 2), 0) / length;
        const advStd = Math.sqrt(advVariance) + 1e-8;

        for (let t = 0; t < length; t++) {
            advantages[t] = (advantages[t] - advMean) / advStd;
        }

        let totalActorLoss = 0;
        let totalCriticLoss = 0;
        let clipCount = 0;
        let klSum = 0;
        let epochs = 3; // standard PPO range is 4-10 over the same rollout
        const GRADIENT_CLIP = 5.0;
        const clipGrad = (g: number) => (isNaN(g) ? 0 : clamp(g, -GRADIENT_CLIP, GRADIENT_CLIP));

        let minibatchesSinceYield = 0;

        for (let e = 0; e < epochs; e++) {
            for (let mbStart = 0; mbStart < length; mbStart += MINIBATCH_SIZE) {
                const batchWidth = Math.min(MINIBATCH_SIZE, length - mbStart);

                // gather this minibatch's states into (inputSize, batchWidth)
                if (!this.stateBatchScratch || this.stateBatchScratch.cols !== batchWidth) {
                    this.stateBatchScratch = new Matrix(this.inputSize, batchWidth);
                }
                const stateBatch = this.stateBatchScratch;
                for (let col = 0; col < batchWidth; col++) {
                    const s = episode[mbStart + col].state;
                    for (let row = 0; row < this.inputSize; row++) {
                        stateBatch.data[row * batchWidth + col] = s[row];
                    }
                }

                // critic: one batched forward + backward
                const criticPred = this.critic.predictBatch(stateBatch); // (1, batchWidth)

                if (!this.criticGradBatchScratch || this.criticGradBatchScratch.cols !== batchWidth) {
                    this.criticGradBatchScratch = new Matrix(1, batchWidth);
                }
                const criticGradBatch = this.criticGradBatchScratch;

                for (let col = 0; col < batchWidth; col++) {
                    const error = criticPred.data[col] - returns[mbStart + col];
                    // same huber-ish gradient clip as the old single-sample train()
                    criticGradBatch.data[col] = Math.max(-1, Math.min(1, error));
                    totalCriticLoss += error * error;
                }

                this.critic.backwardBatchWithGradient(criticGradBatch, this.criticLearningRate);

                // actor: the clipped-surrogate objective is inherently per-sample
                // scalar math (ratio, clipping, log-prob derivatives) — no useful
                // matrix form. what's batched is the expensive part: forward pass
                // producing mean/std for the whole minibatch, and one backward
                // pass applying all gradients together instead of 64 separate ones
                const actorOut = this.actor.predictBatch(stateBatch); // (2, batchWidth): row0=meanRaw, row1=logStdRawUnclamped

                if (!this.actorGradBatchScratch || this.actorGradBatchScratch.cols !== batchWidth) {
                    this.actorGradBatchScratch = new Matrix(2, batchWidth);
                }
                const actorGradBatch = this.actorGradBatchScratch;

                for (let col = 0; col < batchWidth; col++) {
                    const idx = mbStart + col;
                    const { action, logProb: oldLogProb } = episode[idx];
                    const advantage = advantages[idx];

                    const meanRaw = actorOut.data[0 * batchWidth + col];
                    const logStdRawUnclamped = actorOut.data[1 * batchWidth + col];
                    const logStdRaw = clamp(logStdRawUnclamped, LOG_STD_MIN, LOG_STD_MAX);
                    const liveMean = Math.tanh(meanRaw);
                    const liveStd = Math.exp(logStdRaw);

                    const newLogProb = this.calculateLogProb(action, liveMean, liveStd);

                    klSum += (oldLogProb - newLogProb);

                    // exp(A - B) == A/B, so this is newProb/oldProb without leaving log-space
                    const ratio = Math.exp(newLogProb - oldLogProb);

                    const unclippedObjective = ratio * advantage;
                    const clippedObjective = clamp(ratio, 1 - this.clipRatio, 1 + this.clipRatio) * advantage;
                    const objective = Math.min(unclippedObjective, clippedObjective);

                    let dObj_dRatio = 0;
                    if (unclippedObjective === objective) {
                        dObj_dRatio = advantage;
                    } else {
                        dObj_dRatio = 0; // clip active, no gradient from this sample
                        clipCount++;
                    }

                    const dLoss_dRatio = -dObj_dRatio; // maximize objective == minimize -objective
                    const dLoss_dNewLogProb = dLoss_dRatio * ratio; // d(ratio)/d(newLogProb) = ratio

                    const dLogProb_dMean = (action - liveMean) / (liveStd * liveStd);
                    const dLogProb_dLogStdRaw = ((action - liveMean) ** 2) / (liveStd * liveStd) - 1;

                    const dLoss_dMean = dLoss_dNewLogProb * dLogProb_dMean;
                    const dLoss_dLogStdRaw = dLoss_dNewLogProb * dLogProb_dLogStdRaw;

                    const dMean_dMeanRaw = 1 - liveMean * liveMean; // d/dx[tanh(x)]
                    const dLoss_dMeanRaw = clipGrad(dLoss_dMean * dMean_dMeanRaw);
                    const dLoss_dLogStdRawClipped = clipGrad(dLoss_dLogStdRaw);

                    actorGradBatch.data[0 * batchWidth + col] = dLoss_dMeanRaw;
                    actorGradBatch.data[1 * batchWidth + col] = dLoss_dLogStdRawClipped;

                    totalActorLoss += Math.abs(dLoss_dMeanRaw) + Math.abs(dLoss_dLogStdRawClipped);
                }

                // backwardBatchWithGradient averages the gradient across batchWidth
                // internally, i.e. standard mini-batch SGD
                this.actor.backwardBatchWithGradient(actorGradBatch, this.actorLearningRate);

                minibatchesSinceYield++;
                if (minibatchesSinceYield >= 4) {
                    minibatchesSinceYield = 0;
                    await yieldToEventLoop();
                }
            }
        }

        return {
            actorLoss: totalActorLoss / length,
            criticLoss: totalCriticLoss / length,
            avgAdvantage: advantages.reduce((a, b) => a + Math.abs(b), 0) / length,
            clipFraction: clipCount / length, // fraction of samples where the clip was active
            klDivergence: klSum / length
        };
    }

    public toJSON(): string {
        const dump = (net: NeuralNetwork) => net.layers.map(layer => {
            const l = layer as any;
            if (l.weights && l.biases) {
                return { weights: Array.from(l.weights.data), biases: Array.from(l.biases.data) };
            }
            return null;
        });
        return JSON.stringify({ actor: dump(this.actor), critic: dump(this.critic) });
    }

    public loadJSON(jsonString: string): void {
        const parsed = JSON.parse(jsonString);
        const load = (net: NeuralNetwork, data: any[]) => {
            for (let i = 0; i < net.layers.length; i++) {
                const l = net.layers[i] as any;
                const d = data[i];
                if (l.weights && l.biases && d) {
                    for (let j = 0; j < d.weights.length; j++) l.weights.data[j] = d.weights[j];
                    for (let j = 0; j < d.biases.length; j++) l.biases.data[j] = d.biases[j];
                }
            }
        };
        load(this.actor, parsed.actor);
        load(this.critic, parsed.critic);
    }
}