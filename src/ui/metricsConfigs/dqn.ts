import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface DQNMetrics extends BaseMetrics {
    currentLoss: number;
    currentQ: number;
    lossHistory: number[];
    qValueHistory: number[];
    latestQValues: number[];
    currentActionIndex: number;
    epsilon: number;
}

// Matches config.ts THRUST_LEVELS: [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0]
const THRUST_BAR_LABELS = ['-100', '-66', '-33', '0', '33', '66', '100'];

export const dqnUIConfig: AlgorithmUIConfig<DQNMetrics> = {
    extraStatGroups: [
        {
            heading: 'Agent Health (DQN)',
            rows: [
                { label: 'Loss', get: m => m.currentLoss.toFixed(4) },
                { label: 'Avg Max Q', get: m => m.currentQ.toFixed(2) },
                {
                    label: 'Chaos (ε)', get: m => (m.epsilon * 100).toFixed(1) + '%',
                    status: m => m.epsilon < 0.05 ? { text: 'EXPLOITING', kind: 'good' }
                        : m.epsilon > 0.5 ? { text: 'EXPLORING', kind: 'warn' }
                            : null,
                },
            ],
        },
    ],
    extraCharts: [
        {
            title: 'Q-Values per Thrust Level',
            kind: 'bar',
            get: m => m.latestQValues,
            barLabels: THRUST_BAR_LABELS,
            highlightIndex: m => m.currentActionIndex,
        },
        { title: 'Loss History', kind: 'line', color: '#ef4444', get: m => m.lossHistory },
        { title: 'Avg Max Q History', kind: 'line', color: '#8b5cf6', get: m => m.qValueHistory },
    ],
};
