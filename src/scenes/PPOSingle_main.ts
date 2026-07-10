import { CanvasRenderer } from '../renderer/CanvasRenderer';
import type { RenderableEnvironment } from '../renderer/CanvasRenderer';
import { MetricsPanel } from '../ui/MetricsPanel';
import { ppoUIConfig } from '../ui/metricsConfigs/ppo';
import type { PPOMetrics } from '../ui/metricsConfigs/ppo';

const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
if (!canvas) throw new Error('Could not find sim-canvas in DOM');
const sidebar = document.getElementById('sidebar');
if (!sidebar) throw new Error('Could not find sidebar in DOM');

const renderer = new CanvasRenderer(canvas);

const metrics: PPOMetrics = {
    episode: 1, score: 0, stepsThisEpisode: 0, maxSurvivalTime: 0,
    evalSurvivalSeconds: 0, evalMaxSurvivalSeconds: 0,
    scoreHistory: [], survivalTimeHistory: [],
    stepsPerSecond: 0, lastTrainMs: 0, avgTrainMs: 0,
    maxWorkerFrameGapMs: 0, mainThreadFrameGapMs: 0,
    currentCriticLoss: 0, currentAdvantage: 0, currentClipFraction: 0, currentKlDivergence: 0,
    actionHistory: [], currentActionMean: 0, currentActionStd: 0,
    currentThrustNewtons: 0, maxThrustNewtons: 0,
};

const worker = new Worker(new URL('../training/workers/ppoSingle.worker.ts', import.meta.url), { type: 'module' });

let latestFrame: RenderableEnvironment | null = null;
let renderEnabled = true;

const panel = new MetricsPanel(sidebar, metrics, ppoUIConfig, {
    onRenderToggle: enabled => {
        renderEnabled = enabled;
        worker.postMessage({ type: 'setRenderEnabled', enabled });
    },
    onPreviewToggle: live => worker.postMessage({ type: 'setPreviewMode', mode: live ? 'live' : 'training' }),
    onSave: () => worker.postMessage({ type: 'save' }),
    onLoadFile: json => worker.postMessage({ type: 'load', json }),
});

worker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (!msg || typeof msg !== 'object') return;

    switch (msg.type) {
        case 'metrics':
            Object.assign(metrics, msg.payload);
            break;
        case 'frame':
            latestFrame = msg.payload as RenderableEnvironment;
            break;
        case 'brainData': {
            const { json, episode } = msg.payload as { json: string; episode: number };
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ppo-single-pendulum-ep${episode}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            break;
        }
        case 'loaded':
            console.log('Model loaded into the training Worker.');
            break;
    }
};

let lastRenderAt = performance.now();
let maxMainThreadFrameGapMs = 0;

setInterval(() => {
    metrics.mainThreadFrameGapMs = maxMainThreadFrameGapMs;
    maxMainThreadFrameGapMs = 0;
}, 1000);

function renderLoop() {
    const now = performance.now();
    const gap = now - lastRenderAt;
    if (gap > maxMainThreadFrameGapMs) maxMainThreadFrameGapMs = gap;
    lastRenderAt = now;

    panel.update();

    if (renderEnabled && latestFrame) {
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        renderer.render(latestFrame);
    }

    requestAnimationFrame(renderLoop);
}

renderLoop();
