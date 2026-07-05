export class Matrix {
    public rows: number;
    public cols: number;
    public data: Float32Array;

    constructor(rows: number, cols: number) {
        this.rows = rows;
        this.cols = cols;
        // Float32Array initializes all values to 0 automatically.
        // It's a continuous block of memory, making it incredibly fast for CPU cache.
        this.data = new Float32Array(rows * cols);
    }

    // ==========================================
    // INITIALIZATION & FACTORY METHODS
    // ==========================================

    /**
     * Creates a Matrix filled with 0s.
     */
    static zeros(rows: number, cols: number): Matrix {
        return new Matrix(rows, cols);
    }

    /**
     * Creates a Matrix filled with random numbers drawn from a 
     * Standard Normal Distribution (mean 0, variance 1) using the Box-Muller transform.
     */
    static randn(rows: number, cols: number): Matrix {
        const result = new Matrix(rows, cols);
        for (let i = 0; i < result.data.length; i++) {
            let u = 0, v = 0;
            while (u === 0) u = Math.random(); // Converting [0,1) to (0,1)
            while (v === 0) v = Math.random();
            const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            result.data[i] = num;
        }
        return result;
    }

    /**
     * Converts a standard 2D JavaScript array into our highly-optimized Matrix.
     */
    static fromArray(arr: number[][]): Matrix {
        const rows = arr.length;
        const cols = arr[0].length;
        const result = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result.data[i * cols + j] = arr[i][j];
            }
        }
        return result;
    }

    /**
     * Converts the Matrix back to a 2D array (useful for debugging or interfacing).
     */
    public toArray(): number[][] {
        const arr: number[][] = [];
        for (let i = 0; i < this.rows; i++) {
            const row: number[] = [];
            for (let j = 0; j < this.cols; j++) {
                row.push(this.data[i * this.cols + j]);
            }
            arr.push(row);
        }
        return arr;
    }

    // ==========================================
    // MATRIX OPERATIONS (Returns a new Matrix)
    // ==========================================

    /**
     * Standard Matrix Multiplication (Dot Product).
     * O(n^3) complexity, optimized for 1D array indexing.
     */
    static dot(a: Matrix, b: Matrix): Matrix {
        if (a.cols !== b.rows) {
            throw new Error(`Matrix mismatch: Columns of A (${a.cols}) must match Rows of B (${b.rows})`);
        }
        const result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
                }
                result.data[i * b.cols + j] = sum;
            }
        }
        return result;
    }

    /**
     * Flips rows and columns. Critical for calculating gradients during Backpropagation.
     */
    public transpose(): Matrix {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                // Read from [i, j], write to [j, i]
                result.data[j * this.rows + i] = this.data[i * this.cols + j];
            }
        }
        return result;
    }

    // ==========================================
    // ELEMENT-WISE OPERATIONS (In-Place for speed)
    // ==========================================

    public add(n: Matrix | number): this {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) throw new Error("Matrix dimensions must match for addition");
            for (let i = 0; i < this.data.length; i++) this.data[i] += n.data[i];
        } else {
            for (let i = 0; i < this.data.length; i++) this.data[i] += n;
        }
        return this; // Return 'this' to allow method chaining (e.g., m.add(1).mult(2))
    }

    public sub(n: Matrix | number): this {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) throw new Error("Matrix dimensions must match for subtraction");
            for (let i = 0; i < this.data.length; i++) this.data[i] -= n.data[i];
        } else {
            for (let i = 0; i < this.data.length; i++) this.data[i] -= n;
        }
        return this;
    }

    public mult(n: Matrix | number): this {
        if (n instanceof Matrix) {
            // Hadamard Product (Element-wise multiplication)
            if (this.rows !== n.rows || this.cols !== n.cols) throw new Error("Matrix dimensions must match for Hadamard product");
            for (let i = 0; i < this.data.length; i++) this.data[i] *= n.data[i];
        } else {
            // Scalar multiplication
            for (let i = 0; i < this.data.length; i++) this.data[i] *= n;
        }
        return this;
    }

    // ==========================================
    // FUNCTIONAL OPERATIONS
    // ==========================================

    /**
     * Applies a function to every element in the matrix in-place.
     * Perfect for Activation Functions (ReLU, Sigmoid).
     */
    public map(fn: (val: number, index: number) => number): this {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = fn(this.data[i], i);
        }
        return this;
    }
}