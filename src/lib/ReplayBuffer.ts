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

    public add(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
        this.buffer[this.pointer] = { state, action, reward, nextState, done };
        this.pointer = (this.pointer + 1) % this.maxSize;
        this.currentSize = Math.min(this.currentSize + 1, this.maxSize);
    }

    public sample(batchSize: number): Memory[] {
        const batch: Memory[] = [];
        for (let i = 0; i < batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * this.currentSize);
            batch.push(this.buffer[randomIndex]);
        }
        return batch;
    }

    public isReady(batchSize: number): boolean {
        return this.currentSize >= batchSize;
    }

    public clear(): void {
        // no need to wipe the array itself; resetting pointer/size is enough
        // since new writes will overwrite stale entries before they're ever sampled
        this.pointer = 0;
        this.currentSize = 0;
    }
}