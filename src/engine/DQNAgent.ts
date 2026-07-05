import  { DenseLayer } from "../lib/DenseLayer";
import  { NeuralNetwork } from "../lib/NeuralNetwork";
import  { ReLULayer } from "../lib/ReLULayer";
import  { ReplayBuffer } from "../lib/ReplayBuffer";

export class DQNAgent {
    public brain: NeuralNetwork;
    public memory: ReplayBuffer;

    // The Personality (Hyperparameters)
    public epsilon: number = 1.0;           // Starts completely drunk/random
    public epsilonDecay: number = 0.995;    // Sobers up over time
    public epsilonMin: number = 0.01;       // Always retains 1% curiosity
    public gamma: number = 0.99;            // Discount factor (impatience)
    public learningRate: number = 0.001;    // The size of our calculus steps
    public batchSize: number = 32;          // How many memories to dream about at once

    private numActions: number;

    constructor(inputSize: number, numActions: number) {
        this.numActions = numActions;
        this.memory = new ReplayBuffer(10000); // The 10k rolling window

        // ==========================================
        // ASSEMBLING THE DEEP NEURAL NETWORK
        // Architecture: 
        // 1. Input Layer -> 64 Neurons
        // 2. ReLU Activation
        // 3. 64 Neurons -> 64 Neurons
        // 4. ReLU Activation
        // 5. 64 Neurons -> Output Q-Values
        // ==========================================
        this.brain = new NeuralNetwork([
            new DenseLayer(inputSize, 64),
            new ReLULayer(),
            new DenseLayer(64, 64),
            new ReLULayer(),
            new DenseLayer(64, numActions)
        ]);
    }

    /**
     * SENSE & DECIDE: Epsilon-Greedy action selection
     */
    public getAction(state: number[]): number {
        // EXPLORE: Random action
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        }

        // EXPLOIT: Ask the brain
        const qValues = this.brain.predict(state);

        let bestAction = 0;
        let maxQ = qValues[0];
        for (let i = 1; i < this.numActions; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    /**
     * STORE: Add an experience to the Replay Buffer
     */
    public remember(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
        this.memory.add(state, action, reward, nextState, done);
    }

    /**
     * DREAM & LEARN: Returns the average Loss and average Target Q-Value for this batch
     * so we can plot it on our diagnostic dashboard!
     */
    public replay(): { loss: number, qValue: number } | null {
        // Wait until we have at least 32 memories before we start training
        if (!this.memory.isReady(this.batchSize)) {
            return null;
        }

        // 1. Grab a random handful of memories from the past
        const batch = this.memory.sample(this.batchSize);
        let totalLoss = 0;
        let totalTargetQ = 0;

        for (const memory of batch) {
            // 2. What did the network originally think the Q-values were for this state?
            const targetQ = this.brain.predict(memory.state);

            // 3. The Bellman Equation: Update ONLY the specific action we took
            if (memory.done) {
                targetQ[memory.action] = memory.reward; // If we died, the future is bleak
                totalTargetQ += memory.reward;
            } else {
                const nextQ = this.brain.predict(memory.nextState);
                const maxFutureQ = Math.max(...nextQ);
                targetQ[memory.action] = memory.reward + (this.gamma * maxFutureQ);
                totalTargetQ += targetQ[memory.action];
            }

            // 4. THE CALCULUS: Train the network to output this new, slightly better targetQ
            // The brain.train() method now returns the MSE Loss for this step.
            totalLoss += this.brain.train(memory.state, targetQ, this.learningRate);
        }

        return {
            loss: totalLoss / this.batchSize,
            qValue: totalTargetQ / this.batchSize
        };
    }

    /**
     * Called at the end of an episode
     */
    public decayEpsilon(): void {
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }
}