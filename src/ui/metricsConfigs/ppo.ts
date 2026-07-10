import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface PPOMetrics extends BaseMetrics {
    currentCriticLoss: number;
    currentAdvantage: number;
    currentClipFraction: number;
    currentKlDivergence: number;
    actionHistory: number[];
    currentActionMean: number;
    currentActionStd: number;
    currentThrustNewtons: number;
    maxThrustNewtons: number;
}

export const ppoUIConfig: AlgorithmUIConfig<PPOMetrics> = {
    extraStatGroups: [
        {
            heading: 'Actor Distribution',
            rows: [
                { label: 'Mean (μ)', get: m => m.currentActionMean.toFixed(3) },
                { label: 'Std Dev (σ)', get: m => m.currentActionStd.toFixed(3) },
                { label: 'Last Thrust', get: m => m.currentThrustNewtons.toFixed(0) + 'N' },
                { label: 'Max Thrust (±)', get: m => m.maxThrustNewtons.toFixed(0) + 'N' },
            ],
        },
        {
            heading: 'Agent Health (PPO)',
            rows: [
                { label: 'Critic Loss', get: m => m.currentCriticLoss.toFixed(4) },
                {
                    label: '|Advantage|', get: m => m.currentAdvantage.toFixed(2),
                    status: m => m.currentAdvantage < 2.0 ? { text: 'GOOD', kind: 'good' }
                        : m.currentAdvantage < 5.0 ? { text: 'LEARNING', kind: 'warn' }
                            : { text: 'HIGH', kind: 'bad' },
                },
                {
                    label: 'Clip Fraction', get: m => (m.currentClipFraction * 100).toFixed(1) + '%',
                    status: m => m.currentClipFraction < 0.02 ? { text: 'STALLED (Too Low)', kind: 'bad' }
                        : m.currentClipFraction > 0.30 ? { text: 'THRASHING (Too High)', kind: 'bad' }
                            : m.currentClipFraction > 0.20 ? { text: 'FAST', kind: 'warn' }
                                : { text: 'HEALTHY', kind: 'good' },
                },
                {
                    label: 'KL Divergence', get: m => m.currentKlDivergence.toFixed(4),
                    status: m => m.currentKlDivergence < 0.001 ? { text: 'STALLED', kind: 'bad' }
                        : m.currentKlDivergence > 0.08 ? { text: 'DANGER (Collapsing)', kind: 'bad' }
                            : m.currentKlDivergence > 0.03 ? { text: 'UNSTABLE', kind: 'warn' }
                                : { text: 'STABLE', kind: 'good' },
                },
            ],
        },
    ],
    extraCharts: [
        {
            title: 'Action Distribution (Last 30)',
            kind: 'histogramWithGaussian',
            get: m => m.actionHistory,
            gaussianMean: m => m.currentActionMean,
            gaussianStd: m => m.currentActionStd,
        },
    ],
};
