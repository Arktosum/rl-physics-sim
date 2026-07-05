import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class ReLULayer implements Layer {
    private input: Matrix | null = null;

    /**
     * FORWARD PASS
     * Applies the ReLU function: f(x) = max(0, x)
     */
    forward(input: Matrix): Matrix {
        // Save the input so we know which neurons "fired" for the backward pass
        this.input = input;

        const output = new Matrix(input.rows, input.cols);
        for (let i = 0; i < input.data.length; i++) {
            output.data[i] = Math.max(0, input.data[i]);
        }

        return output;
    }

    /**
     * BACKWARD PASS
     * Chain Rule: inputGradient = outputGradient * derivative_of_ReLU(input)
     */
    backward(outputGradient: Matrix, learningRate: number): Matrix {
        if (!this.input) {
            throw new Error("Must call forward() before backward() can calculate gradients.");
        }

        const inputGradient = new Matrix(outputGradient.rows, outputGradient.cols);

        for (let i = 0; i < outputGradient.data.length; i++) {
            // The derivative of ReLU is 1 if input > 0, and 0 otherwise.
            // So we only let the gradient flow backward if the neuron originally fired.
            inputGradient.data[i] = this.input.data[i] > 0 ? outputGradient.data[i] : 0;
        }

        // Note: We don't use learningRate here because activation layers have no learnable weights!
        return inputGradient;
    }
}