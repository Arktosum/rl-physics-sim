import { Environment } from './engine/Environment';
import { CanvasRenderer } from './renderer/CanvasRenderer';
import { PointMass } from './state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from './engine/Constraint';
import { Actuator } from './engine/Actuator';
import { DQNAgent } from './engine/DQNAgent';

// 1. Initialize Canvas
const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = 800;
canvas.height = 600;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

// 2. RESTORE THE CUSTOM PHYSICS ENVIRONMENT!
const env = new Environment(9.81);
const trackHeight = 400;

const cart = new PointMass(400, trackHeight, 10, false);
const pole1 = new PointMass(400, 300, 2, false);
const pole2 = new PointMass(400, 200, 2, false);

const link1 = new DistanceConstraint(cart, pole1, 100);
const link2 = new DistanceConstraint(pole1, pole2, 100);
const track = new AxisConstraint(cart, trackHeight);
const screenBounds = new BoundaryConstraint(cart, canvas.width, canvas.height, 0);

env.addPoint(cart);
env.addPoint(pole1);
env.addPoint(pole2);
env.addConstraint(link1);
env.addConstraint(link2);
env.addConstraint(track);
env.addConstraint(screenBounds);

const motor = new Actuator(cart, 1500);

// 3. Initialize Agent (5 inputs for double pendulum, 2 actions)
const agent = new DQNAgent(5, 2);

// Diagnostic Histories
let episode = 1;
let score = 0;
let maxScore = 0;
const scoreHistory: number[] = [];
const lossHistory: number[] = [];
const qValueHistory: number[] = [];

let currentLoss = 0;
let currentQ = 0;

function resetWorld() {
    cart.position.x = 400;
    cart.position.y = trackHeight;
    cart.oldPosition.x = 400;
    cart.oldPosition.y = trackHeight;

    const wobble1 = (Math.random() - 0.5) * 0.3;
    const wobble2 = (Math.random() - 0.5) * 0.3;

    pole1.position.x = 400 + Math.sin(wobble1) * 100;
    pole1.position.y = trackHeight - Math.cos(wobble1) * 100;
    pole1.oldPosition.x = pole1.position.x;
    pole1.oldPosition.y = pole1.position.y;

    pole2.position.x = pole1.position.x + Math.sin(wobble1 + wobble2) * 100;
    pole2.position.y = pole1.position.y - Math.cos(wobble1 + wobble2) * 100;
    pole2.oldPosition.x = pole2.position.x;
    pole2.oldPosition.y = pole2.position.y;

    motor.activeDirection = 0;
}

// THE MOST IMPORTANT FIX: NORMALIZATION
// Neural networks explode if you feed them raw pixel coordinates like "700".
// Everything must be scaled to roughly between -1.0 and 1.0.
function senseAndNormalize(dt: number): number[] {
    const dx1 = pole1.position.x - cart.position.x;
    const dy1 = cart.position.y - pole1.position.y;
    const dx2 = pole2.position.x - pole1.position.x;
    const dy2 = pole1.position.y - pole2.position.y;

    const a1 = Math.atan2(dx1, dy1);
    const a2 = Math.atan2(dx2, dy2);

    const oldDx1 = pole1.oldPosition.x - cart.oldPosition.x;
    const oldDy1 = cart.oldPosition.y - pole1.oldPosition.y;
    const oldDx2 = pole2.oldPosition.x - pole1.oldPosition.x;
    const oldDy2 = pole1.oldPosition.y - pole2.oldPosition.y;

    const oldA1 = Math.atan2(oldDx1, oldDy1);
    const oldA2 = Math.atan2(oldDx2, oldDy2);

    const v1 = (a1 - oldA1) / dt;
    const v2 = (a2 - oldA2) / dt;

    return [
        (cart.position.x - 400) / 400, // Normalized X: roughly [-1, 1]
        a1 / 1.0,                      // Normalized Angle 1
        a2 / 1.0,                      // Normalized Angle 2
        v1 / 10.0,                     // Normalized Velocity 1
        v2 / 10.0                      // Normalized Velocity 2
    ];
}

const FIXED_DT = 0.016;
resetWorld();
let currentState = senseAndNormalize(FIXED_DT);

// Since Deep Learning requires heavy calculus matrix multiplications,
// doing 50 steps per frame will freeze the browser. 2 steps keeps it 
// visually smooth while allowing the neural network to train.
const STEPS_PER_FRAME = 2;

function step() {
    for (let i = 0; i < STEPS_PER_FRAME; i++) {

        const action = agent.getAction(currentState);
        motor.activeDirection = action === 0 ? -1 : 1;
        motor.apply();
        env.update(FIXED_DT);

        const nextState = senseAndNormalize(FIXED_DT);
        const rawX = cart.position.x;
        const rawA1 = nextState[1] * 1.0;
        const rawA2 = nextState[2] * 1.0;

        const isDead = Math.abs(rawA1) > 0.8 || Math.abs(rawA2) > 0.8 || rawX < 50 || rawX > 750;

        // Shaped reward: fight to stay completely upright
        const reward = isDead ? -100 : (0.5 * Math.cos(rawA1) + 0.5 * Math.cos(rawA2));

        agent.remember(currentState, action, reward, nextState, isDead);

        // Train and capture metrics
        const metrics = agent.replay();
        if (metrics) {
            currentLoss = currentLoss * 0.99 + metrics.loss * 0.01; // Smooth it out
            currentQ = currentQ * 0.99 + metrics.qValue * 0.01;

            // Record every few steps to avoid giant arrays slowing down canvas draw
            if (Math.random() < 0.05) {
                lossHistory.push(currentLoss);
                qValueHistory.push(currentQ);
                if (lossHistory.length > 200) lossHistory.shift();
                if (qValueHistory.length > 200) qValueHistory.shift();
            }
        }

        if (isDead) {
            if (score > maxScore) maxScore = score;
            scoreHistory.push(score);
            if (scoreHistory.length > 200) scoreHistory.shift();

            agent.decayEpsilon();
            resetWorld();
            episode++;
            score = 0;
            currentState = senseAndNormalize(FIXED_DT);
        } else {
            score += reward;
            currentState = nextState;
        }
    }

    renderer.render(env);
    drawDiagnostics();
    requestAnimationFrame(step);
}

// ==========================================
// ADVANCED DIAGNOSTICS UI
// ==========================================
function drawDiagnostics() {
    // 1. Text Stats
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(10, 10, 250, 180);
    ctx.fillStyle = 'white';
    ctx.font = '14px monospace';
    ctx.fillText(`Episode:  ${episode}`, 20, 35);
    ctx.fillText(`Score:    ${score.toFixed(1)}`, 20, 55);
    ctx.fillText(`MaxScore: ${maxScore.toFixed(1)}`, 20, 75);
    ctx.fillText(`Chaos(E): ${(agent.epsilon * 100).toFixed(1)}%`, 20, 95);

    // Neural Net specific stats
    ctx.fillStyle = '#ef4444'; // Red
    ctx.fillText(`MSE Loss: ${currentLoss.toFixed(4)}`, 20, 130);
    ctx.fillStyle = '#8b5cf6'; // Purple
    ctx.fillText(`Avg Q-Val: ${currentQ.toFixed(2)}`, 20, 150);

    // 2. Draw Mini-Charts
    drawChart(lossHistory, canvas.width - 220, 20, 200, 60, 'MSE Loss', '#ef4444');
    drawChart(qValueHistory, canvas.width - 220, 90, 200, 60, 'Avg Max Q', '#8b5cf6');
    drawChart(scoreHistory, canvas.width - 220, 160, 200, 60, 'Scores', '#4ade80');
}

function drawChart(data: number[], x: number, y: number, w: number, h: number, label: string, color: string) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = '#4b5563';
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = 'white';
    ctx.font = '10px monospace';
    ctx.fillText(label, x + 5, y + 12);

    if (data.length < 2) return;

    // Auto-scale Y axis
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = (max - min) || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    const stepX = w / (data.length - 1);
    for (let i = 0; i < data.length; i++) {
        const px = x + i * stepX;
        const py = y + h - ((data[i] - min) / range) * h;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();
}

requestAnimationFrame(step);