export class QLearningAgent {
    private qTable: Map<string, number[]>;

    // How many (actions) are available at every state? (For Cart-Pole, it's 2)
    private numActions: number;

    // 2. The Personality (Hyperparameters)
    private alpha: number;          // Learning Rate (The stubborn eraser)
    private gamma: number;          // Discount Factor (The impatience)
    private epsilon: number;        // Exploration Rate (The wooden coin probability)
    private epsilonDecay: number;   // The Sobbering Factor (How fast we stop being drunk)


    constructor(numActions: number) {
        // When the agent is born, we hand it a completely blank notebook
        this.qTable = new Map<string, number[]>();
        this.numActions = numActions;

        // Hardcoding our personality traits:
        this.alpha = 0.1;           // Shift our beliefs by exactly 10% on every step
        this.gamma = 0.99;          // Highly patient, but still bounded to avoid infinity
        this.epsilon = 1.0;         // Start out 100% drunk! Explore everything at the beginning and slowly sober up!
        this.epsilonDecay = 0.995;  // Every time a full life ends, multiply epsilon by this to sober up

    }

    /**
     * Helper Method: Open the notebook to a specific page.
     * If the state has never been visited, create a new blank page.
     */
    public getQValues(state: string): number[] {
        // Check if the page exists
        if (!this.qTable.has(state)) {

            // It doesn't exist! Create an array filled with 0s.
            // If numActions is 2, this makes [0, 0]
            const blankPage = new Array(this.numActions).fill(0);

            // Write this blank page into the notebook
            this.qTable.set(state, blankPage);
        }

        // Return the numbers written on that page
        return this.qTable.get(state)!;
    }

    /**
     * 3. The Decider (Epsilon-Greedy Policy)
     * Takes the current state and returns an action.
     */
    public getAction(state: string): number {
        
        // Roll the 100-sided die (Math.random() gives a float between 0 and 1)
        if (Math.random() < this.epsilon) {
            // EXPLORE: Act like the drunk guy. 
            // Ignore the notebook and pick a completely random action.
            return Math.floor(Math.random() * this.numActions);
        } 
        
        // EXPLOIT: Act like the King. 
        // We rolled higher than epsilon, so we look at the notebook.
        const qValues = this.getQValues(state);
        
        // Find the action with the highest expected reward (the max Q-value)
        let bestAction = 0;
        let maxQ = qValues[0];

        for (let i = 1; i < this.numActions; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        
        // Confidently take the best action
        return bestAction;
    }

    /**
     * 4. The Learner (The Bellman Update)
     * Takes the story of what just happened and updates the notebook.
     */
    public learn(state: string, action: number, reward: number, nextState: string, done: boolean): void {
        
        // 1. Open the notebook to today's page and look at the old belief
        const qValues = this.getQValues(state);
        const oldBelief = qValues[action];

        // 2. Look ahead to tomorrow
        let maxFutureQ = 0;
        
        // CRITICAL CHECK: Is the agent dead? 
        if (!done) {
            // If the game isn't over, assume the future self is a Flawless God (max)
            const nextQValues = this.getQValues(nextState);
            maxFutureQ = Math.max(...nextQValues); 
        }
        // (If the agent is dead, there is no tomorrow, so maxFutureQ stays 0)

        // 3. Calculate the absolute truth (The Target)
        // Target = Immediate Reward + (Discount Factor * Best Future Q-Value)
        const target = reward + (this.gamma * maxFutureQ);

        // 4. Calculate the Error (Reality vs. Expectation)
        const error = target - oldBelief;

        // 5. Use the stubborn eraser (Alpha) to update the notebook
        qValues[action] = oldBelief + (this.alpha * error);
    }

}