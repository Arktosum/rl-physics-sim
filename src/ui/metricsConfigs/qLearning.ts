import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface QLearningMetrics extends BaseMetrics {
    statesVisited: number;
    coverageFraction: number; // 0..1, statesVisited / totalPossibleStates
    singleVisitFraction: number; // 0..1, fraction of visit-counts that are exactly 1 (noisy Q-estimates)
    totalVisits: number;
}

export const qLearningUIConfig: AlgorithmUIConfig<QLearningMetrics> = {
    extraStatGroups: [
        {
            heading: 'Agent Health (Q-Learning)',
            rows: [
                // No epsilon/"chaos" row: this agent runs UCB exploration by
                // default (docs/journey/01-q-learning.md), not epsilon-greedy —
                // epsilon is genuinely inert in that mode, so showing it would
                // just be a number that doesn't reflect what's actually
                // happening. Coverage is the real signal for this algorithm.
                { label: 'States Visited', get: m => m.statesVisited.toLocaleString() },
                { label: 'Total Visits', get: m => m.totalVisits.toLocaleString() },
                {
                    label: 'Coverage', get: m => (m.coverageFraction * 100).toFixed(1) + '%',
                    // Coverage growing much slower than table size as bins get
                    // finer is the actual ceiling of tabular Q-learning — worth
                    // seeing directly, not just in the write-up.
                    status: m => m.coverageFraction < 0.05 ? { text: 'SPARSE', kind: 'bad' }
                        : m.coverageFraction < 0.2 ? { text: 'GROWING', kind: 'warn' }
                            : { text: 'GOOD', kind: 'good' },
                },
                {
                    label: 'Single-Visit States', get: m => (m.singleVisitFraction * 100).toFixed(1) + '%',
                    status: m => m.singleVisitFraction > 0.5 ? { text: 'NOISY ESTIMATES', kind: 'warn' } : null,
                },
            ],
        },
    ],
    extraCharts: [],
};
