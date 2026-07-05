import { Matrix } from './Matrix';
import type { Layer } from './Layer';

export class NeuralNetwork {
    public layers: Layer[];

    constructor(layers: Layer[]) {
        this.layers = layers;
    }

    /**
     * FORWARD PASS
     * Takes an array of raw numbers (e.g. physics state), passes it through every layer,
     * and returns the final prediction as an array of numbers (e.g. Q-Values).
     */
    public predict(inputArray: number[]): number[] {
        // 1. Convert the flat input array into a column Matrix (Rows = length, Cols = 1)
        let currentData = new Matrix(inputArray.length, 1);
        for (let i = 0; i < inputArray.length; i++) {
            currentData.data[i] = inputArray[i];
        }

        // 2. Push the data forward sequentially through the Lego bricks
        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        // 3. Convert the final column Matrix back into a flat JavaScript array
        return Array.from(currentData.data);
    }

    /**
     * BACKWARD PASS (The Learning Step)
     * Returns the Mean Squared Error (Loss) for diagnostic tracking.
     */
    public train(inputArray: number[], targetArray: number[], learningRate: number): number {
        // --- STEP 1: Forward Pass ---
        let currentData = new Matrix(inputArray.length, 1);
        for (let i = 0; i < inputArray.length; i++) {
            currentData.data[i] = inputArray[i];
        }

        for (const layer of this.layers) {
            currentData = layer.forward(currentData);
        }

        // --- STEP 2: Calculate the Initial Error Gradient & Loss ---
        let gradient = new Matrix(targetArray.length, 1);
        let totalError = 0;

        for (let i = 0; i < targetArray.length; i++) {
            const error = currentData.data[i] - targetArray[i];
            gradient.data[i] = error;
            totalError += error * error; // Squared Error
        }

        const mseLoss = totalError / targetArray.length;

        // --- STEP 3: Backward Pass ---
        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(gradient, learningRate);
        }

        return mseLoss;
    }
}