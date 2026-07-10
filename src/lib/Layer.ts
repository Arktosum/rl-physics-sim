import { Matrix } from './Matrix';

export interface Layer {
    // must internally cache whatever state (e.g. input) backward() needs
    forward(input: Matrix): Matrix;

    backward(outputGradient: Matrix, learningRate: number): Matrix;

    /**
     * Optional batched variants: process a (features, batchWidth) matrix — one
     * column per sample — as a handful of wide matmuls instead of looping
     * forward()/backward() once per sample. Only worth implementing for
     * layers with learnable parameters whose gradient needs to be SUMMED
     * across the batch (e.g. DenseLayer); parameter-free layers like
     * ReLULayer are already shape-agnostic and can be fed a batch straight
     * through forward()/backward() with no changes. NeuralNetwork falls back
     * to the single-sample methods for any layer that doesn't implement these.
     */
    forwardBatch?(input: Matrix): Matrix;
    backwardBatch?(outputGradient: Matrix, learningRate: number): Matrix;
}