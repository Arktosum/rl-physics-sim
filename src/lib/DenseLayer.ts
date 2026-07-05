import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class DenseLayer implements Layer {
    private weights: Matrix;
    private biases: Matrix;
    private input: Matrix | null = null;

    constructor(inputSize: number, outputSize: number) {
        // He Initialization: Random normal values scaled by sqrt(2 / inputSize).
        // This stops our gradients from vanishing into 0 or exploding into Infinity.
        this.weights = Matrix.randn(outputSize, inputSize).mult(Math.sqrt(2 / inputSize));

        // Biases safely start at 0
        this.biases = Matrix.zeros(outputSize, 1);
    }

    /**
     * FORWARD PASS
     * Math: Output = (Weights • Input) + Biases
     */
    forward(input: Matrix): Matrix {
        // Save the input so we can use it during Backpropagation later
        this.input = input;

        // Matrix.dot() returns a new Matrix, so .add() modifies that new matrix safely
        return Matrix.dot(this.weights, input).add(this.biases);
    }

    /**
     * BACKWARD PASS (The Calculus)
     * Takes the error gradient from the layer ahead of it, updates its own weights/biases,
     * and passes the remaining error gradient backward to the layer behind it.
     */
    backward(outputGradient: Matrix, learningRate: number): Matrix {
        if (!this.input) {
            throw new Error("Must call forward() before backward() can calculate gradients.");
        }

        // 1. Calculate Weight Gradients: dW = outputGradient • input^T
        const weightsGradient = Matrix.dot(outputGradient, this.input.transpose());

        // 2. Calculate Input Gradients to pass backward: dX = W^T • outputGradient
        // (We must calculate this BEFORE we mutate the weights in the next step!)
        const inputGradient = Matrix.dot(this.weights.transpose(), outputGradient);

        // 3. Update Weights: W = W - (dW * learningRate)
        weightsGradient.mult(learningRate);
        this.weights.sub(weightsGradient);

        // 4. Update Biases: B = B - (dB * learningRate)
        // The gradient of the bias is exactly equal to the outputGradient.
        // We do this in a raw loop for maximum CPU cache performance.
        for (let i = 0; i < this.biases.data.length; i++) {
            this.biases.data[i] -= outputGradient.data[i] * learningRate;
        }

        // Return the input gradient so the layer behind us can do its own backprop
        return inputGradient;
    }
}