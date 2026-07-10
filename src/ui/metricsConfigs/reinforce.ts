import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface ReinforceMetrics extends BaseMetrics {
    currentCriticLoss: number;
    currentAdvantage: number;
    currentEntropy: number;
    currentGradientClipRate: number;
    actionHistory: number[];
    currentMean: number;
    currentStd: number;
}

export const reinforceUIConfig: AlgorithmUIConfig<ReinforceMetrics> = {
    extraStatGroups: [
        {
            heading: 'Actor Distribution',
            rows: [
                { label: 'Mean (μ)', get: m => m.currentMean.toFixed(3) },
                { label: 'Std Dev (σ)', get: m => m.currentStd.toFixed(3) },
            ],
        },
        {
            heading: 'Agent Health (REINFORCE)',
            rows: [
                { label: 'Critic Loss', get: m => m.currentCriticLoss.toFixed(4) },
                { label: '|Advantage|', get: m => m.currentAdvantage.toFixed(2) },
                {
                    label: 'Policy Entropy', get: m => m.currentEntropy.toFixed(3),
                    // Entropy collapsing toward the LOG_STD_MIN floor (~-1.58 for this
                    // project's std clamp) is the exact failure mode documented in
                    // docs/journey/02-reinforce.md — total policy overconfidence.
                    status: m => m.currentEntropy < -1.4 ? { text: 'COLLAPSING', kind: 'bad' }
                        : m.currentEntropy < -0.5 ? { text: 'NARROWING', kind: 'warn' }
                            : { text: 'HEALTHY', kind: 'good' },
                },
                {
                    label: 'Gradient Clip Rate', get: m => (m.currentGradientClipRate * 100).toFixed(1) + '%',
                    status: m => m.currentGradientClipRate > 0.3 ? { text: 'EXPLODING', kind: 'bad' } : null,
                },
            ],
        },
    ],
    extraCharts: [
        {
            title: 'Action Distribution (Last 200)',
            kind: 'histogramWithGaussian',
            get: m => m.actionHistory,
            gaussianMean: m => m.currentMean,
            gaussianStd: m => m.currentStd,
        },
    ],
};
