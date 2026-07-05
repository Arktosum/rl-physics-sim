export interface Memory {
    state: number[];
    action: number;
    reward: number;
    nextState: number[];
    done: boolean;
}

export class ReplayBuffer {
    private buffer: Memory[];
    private maxSize: number;

    // Using a circular pointer is infinitely faster than using array.shift()
    private pointer: number = 0;
    private currentSize: number = 0;

    constructor(maxSize: number = 10000) {
        this.maxSize = maxSize;
        this.buffer = new Array(maxSize);
    }

    /**
     * 1 & 4. Add the newest memory, automatically overwriting the oldest if full.
     */
    public add(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
        this.buffer[this.pointer] = { state, action, reward, nextState, done };

        // Move pointer forward, loop back to 0 if we hit the max size
        this.pointer = (this.pointer + 1) % this.maxSize;

        // Keep track of how many memories we actually have
        this.currentSize = Math.min(this.currentSize + 1, this.maxSize);
    }

    /**
     * 3. Shuffle and pick a mini-batch of memories
     */
    public sample(batchSize: number): Memory[] {
        const batch: Memory[] = [];
        for (let i = 0; i < batchSize; i++) {
            // Pick a random index from the currently filled portion of the buffer
            const randomIndex = Math.floor(Math.random() * this.currentSize);
            batch.push(this.buffer[randomIndex]);
        }
        return batch;
    }

    /**
     * 2. Check if we have enough memories to start training
     */
    public isReady(batchSize: number): boolean {
        return this.currentSize >= batchSize;
    }
}