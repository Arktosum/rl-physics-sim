export class QLearningAgent {
    private qTable: Map<string, number[]>;
    private visitCounts: Map<string, number[]>;

    // How many (actions) are available at every state? (For Cart-Pole, it's 2)
    private numActions: number;

    // 2. The Personality (Hyperparameters)
    private alpha: number;          // Base learning rate (used if useAdaptiveAlpha = false)
    private gamma: number;          // Discount Factor (The impatience)
    private epsilon: number;        // Exploration Rate (The wooden coin probability)
    private epsilonDecay: number;   // The Sobbering Factor (How fast we stop being drunk)

    // Toggles for the new stuff — flip these to compare strategies
    private useAdaptiveAlpha: boolean = true;   // Robbins-Monro style 1/(1+visits) learning rate
    private useUCB: boolean = true;             // Count-based exploration instead of epsilon-greedy
    private ucbC: number = 4.0;                 // UCB exploration strength — higher = more curious
    private totalSteps: number = 0;             // Needed for the UCB log(totalVisits) term

    constructor(numActions: number) {
        // When the agent is born, we hand it a completely blank notebook
        this.qTable = new Map<string, number[]>();
        this.visitCounts = new Map<string, number[]>();
        this.numActions = numActions;

        // Hardcoding our personality traits:
        this.alpha = 0.1;
        this.gamma = 0.99;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.995;
    }

    /**
     * Helper Method: Open the notebook to a specific page.
     * If the state has never been visited, create a new blank page.
     */
    public getQValues(state: string): number[] {
        if (!this.qTable.has(state)) {
            const blankPage = new Array(this.numActions).fill(0);
            this.qTable.set(state, blankPage);
        }
        return this.qTable.get(state)!;
    }

    /**
     * Helper Method: Open the visit log for a state.
     * Tracks how many times each action has been taken from this state.
     */
    public getVisits(state: string): number[] {
        if (!this.visitCounts.has(state)) {
            this.visitCounts.set(state, new Array(this.numActions).fill(0));
        }
        return this.visitCounts.get(state)!;
    }

    /**
     * 3. The Decider
     * Either epsilon-greedy, or UCB (count-based curiosity) depending on useUCB.
     */
    public getAction(state: string): number {
        const qValues = this.getQValues(state);
        const visits = this.getVisits(state);

        // Hard floor: never let an action go unexplored below N tries,
        // regardless of how far behind its Q-value looks.
        const minVisits = 20;
        for (let i = 0; i < this.numActions; i++) {
            if (visits[i] < minVisits) return i;
        }

        if (this.useUCB) {
            // UCB: prefer actions that are either high-value OR under-explored.
            // Bonus shrinks as an action gets visited more, and grows slowly
            // as total experience grows (so early on, everything looks worth trying).
            const visits = this.getVisits(state);
            let bestAction = 0;
            let bestScore = -Infinity;

            for (let i = 0; i < this.numActions; i++) {
                // If an action has literally never been tried from this state,
                // always try it first — infinite bonus, no need to estimate it.
                if (visits[i] === 0) {
                    bestAction = i;
                    bestScore = Infinity;
                    continue;
                }
                if (bestScore === Infinity) continue; // already found an untried action

                const bonus = this.ucbC * Math.sqrt(Math.log(this.totalSteps + 1) / visits[i]);
                const score = qValues[i] + bonus;

                if (score > bestScore) {
                    bestScore = score;
                    bestAction = i;
                }
            }
            return bestAction;
        }

        // Fallback: original epsilon-greedy behavior
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.numActions);
        }

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
     * 4. The Learner (The Bellman Update)
     */
    public learn(state: string, action: number, reward: number, nextState: string, done: boolean): void {
        const qValues = this.getQValues(state);
        const oldBelief = qValues[action];

        let maxFutureQ = 0;
        if (!done) {
            const nextQValues = this.getQValues(nextState);
            maxFutureQ = Math.max(...nextQValues);
        }

        const target = reward + (this.gamma * maxFutureQ);
        const error = target - oldBelief;

        // Update visit counts BEFORE computing alpha, so the very first
        // visit to a state uses alpha = 1 (fully trust the first observation).
        const visits = this.getVisits(state);
        visits[action]++;
        this.totalSteps++;

        const effectiveAlpha = this.useAdaptiveAlpha
            ? 1 / (1 + visits[action])
            : this.alpha;

        qValues[action] = oldBelief + (effectiveAlpha * error);
    }

    /**
     * Diagnostic: how much of the reachable state space have we actually touched,
     * and how lopsided is that coverage? Call this every few hundred episodes
     * from main.ts and log it — this is your sparsity answer.
     */
    public getCoverageStats(totalPossibleStates?: number) {
        const allVisits: number[] = [];
        for (const counts of this.visitCounts.values()) {
            for (const c of counts) allVisits.push(c);
        }

        const visitedOnce = allVisits.filter(c => c === 1).length;
        const totalVisits = allVisits.reduce((a, b) => a + b, 0);
        const topStates = this.getTopStates(5); // top 5 most-visited states
        return {
            statesVisited: this.qTable.size,
            coverageFraction: totalPossibleStates ? this.qTable.size / totalPossibleStates : undefined,
            totalVisits,
            maxVisits: allVisits.length ? Math.max(...allVisits) : 0,
            singleVisitFraction: allVisits.length ? visitedOnce / allVisits.length : 0,
            topStates: topStates
        };
    }

    public getTopStates(n: number = 10) {
        const entries = Array.from(this.visitCounts.entries())
            .map(([state, visits]) => ({
                state,
                totalVisits: visits.reduce((a, b) => a + b, 0),
                visits,
                qValues: this.qTable.get(state)
            }))
            .sort((a, b) => b.totalVisits - a.totalVisits)
            .slice(0, n);
        return entries;
    }
}