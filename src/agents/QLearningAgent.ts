export class QLearningAgent {
    private qTable: Map<string, number[]>;
    private visitCounts: Map<string, number[]>;

    private numActions: number;

    private alpha: number;          // base learning rate, used only if useAdaptiveAlpha = false
    private gamma: number;          // discount factor
    private epsilon: number;        // exploration rate, used only if useUCB = false

    // flip these to compare exploration/learning-rate strategies
    private useAdaptiveAlpha: boolean = true;   // Robbins-Monro style 1/(1+visits) learning rate
    private useUCB: boolean = true;             // count-based exploration instead of epsilon-greedy
    private ucbC: number = 4.0;                 // UCB exploration strength, higher = more curious
    private totalSteps: number = 0;             // for the UCB log(totalVisits) term

    constructor(numActions: number) {
        this.qTable = new Map<string, number[]>();
        this.visitCounts = new Map<string, number[]>();
        this.numActions = numActions;

        this.alpha = 0.1;
        this.gamma = 0.99;
        this.epsilon = 1.0;
    }

    public getQValues(state: string): number[] {
        if (!this.qTable.has(state)) {
            const blankPage = new Array(this.numActions).fill(0);
            this.qTable.set(state, blankPage);
        }
        return this.qTable.get(state)!;
    }

    public getVisits(state: string): number[] {
        if (!this.visitCounts.has(state)) {
            this.visitCounts.set(state, new Array(this.numActions).fill(0));
        }
        return this.visitCounts.get(state)!;
    }

    public getAction(state: string): number {
        const qValues = this.getQValues(state);
        const visits = this.getVisits(state);

        // hard floor: force every action to be tried at least this many times
        // before trusting its Q-value, regardless of exploration strategy
        const minVisits = 20;
        for (let i = 0; i < this.numActions; i++) {
            if (visits[i] < minVisits) return i;
        }

        if (this.useUCB) {
            // bonus shrinks with visits, grows slowly with total steps —
            // everything looks worth trying early on
            const visits = this.getVisits(state);
            let bestAction = 0;
            let bestScore = -Infinity;

            for (let i = 0; i < this.numActions; i++) {
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

    // pure argmax-Q, no exploration bonus or visit floor — for demoing the
    // current table's preference. UCB has no single "off" switch like epsilon
    // does, hence this separate lookup (mirrors DQNAgent/PPOAgent getGreedyAction)
    public getGreedyAction(state: string): number {
        const qValues = this.getQValues(state);
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

        // increment visits before computing alpha so the first visit to a
        // state uses alpha = 1 (fully trust the first observation)
        const visits = this.getVisits(state);
        visits[action]++;
        this.totalSteps++;

        const effectiveAlpha = this.useAdaptiveAlpha
            ? 1 / (1 + visits[action])
            : this.alpha;

        qValues[action] = oldBelief + (effectiveAlpha * error);
    }

    // fraction of reachable state space actually touched, and how lopsided
    // that coverage is — call periodically from main.ts to gauge sparsity
    public getCoverageStats(totalPossibleStates?: number) {
        const allVisits: number[] = [];
        for (const counts of this.visitCounts.values()) {
            for (const c of counts) allVisits.push(c);
        }

        const visitedOnce = allVisits.filter(c => c === 1).length;
        const totalVisits = allVisits.reduce((a, b) => a + b, 0);
        const topStates = this.getTopStates(5);
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

    public toJSON(): string {
        return JSON.stringify({
            numActions: this.numActions,
            qTable: Array.from(this.qTable.entries()),
            visitCounts: Array.from(this.visitCounts.entries()),
            totalSteps: this.totalSteps,
        });
    }

    public loadJSON(jsonString: string): void {
        const parsed = JSON.parse(jsonString);
        this.qTable = new Map(parsed.qTable);
        this.visitCounts = new Map(parsed.visitCounts);
        this.totalSteps = parsed.totalSteps ?? 0;
    }
}