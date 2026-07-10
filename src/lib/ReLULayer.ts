import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class ReLULayer implements Layer {
    private input: Matrix | null = null;

    // Unlike DenseLayer, ReLULayer's constructor doesn't know its size (it's
    // just whatever shape flows through it), so these are allocated lazily on
    // first use instead of in the constructor, then reused every call after
    // that — same GC-avoidance reasoning as DenseLayer's scratch buffers.
    private outputScratch: Matrix | null = null;
    private inputGradientScratch: Matrix | null = null;

    forward(input: Matrix): Matrix {
        // stashed for backward() to know which neurons fired
        this.input = input;

        if (!this.outputScratch || this.outputScratch.rows !== input.rows || this.outputScratch.cols !== input.cols) {
            this.outputScratch = new Matrix(input.rows, input.cols);
        }
        for (let i = 0; i < input.data.length; i++) {
            this.outputScratch.data[i] = Math.max(0, input.data[i]);
        }

        return this.outputScratch;
    }

    backward(outputGradient: Matrix, _learningRate: number): Matrix {
        if (!this.input) {
            throw new Error("Must call forward() before backward() can calculate gradients.");
        }

        if (!this.inputGradientScratch || this.inputGradientScratch.rows !== outputGradient.rows || this.inputGradientScratch.cols !== outputGradient.cols) {
            this.inputGradientScratch = new Matrix(outputGradient.rows, outputGradient.cols);
        }

        for (let i = 0; i < outputGradient.data.length; i++) {
            this.inputGradientScratch.data[i] = this.input.data[i] > 0 ? outputGradient.data[i] : 0;
        }

        // learningRate unused: activation layers have no learnable weights
        return this.inputGradientScratch;
    }
}