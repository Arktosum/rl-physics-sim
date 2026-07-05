import { Environment } from './engine/Environment';
import { CanvasRenderer } from './renderer/CanvasRenderer';
import { PointMass } from './state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from './engine/Constraint';
import { Actuator } from './engine/Actuator';
import { QLearningAgent } from './engine/QLearningAgent';

// 1. Initialize Canvas and Renderer
const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = 800;
canvas.height = 600;
const ctx = canvas.getContext('2d')!;
const renderer = new CanvasRenderer(canvas);

// Input for STEPS_PER_FRAME
const inputContainer = document.createElement('div');
inputContainer.style.position = 'absolute';
inputContainer.style.top = '170px';
inputContainer.style.left = '20px';
inputContainer.style.zIndex = '10';
inputContainer.innerHTML = `
    <label style="color: white; font-family: monospace;">Steps/Frame: </label>
    <input type="number" id="stepsInput" value="50" style="width: 50px; background: #333; color: white; border: 1px solid #555;">
`;
document.body.appendChild(inputContainer);
const stepsInput = document.getElementById('stepsInput') as HTMLInputElement;

// 2. Initialize Physics Environment
const env = new Environment(9.81);

// 3. Populate our world: Double Pendulum Cart-Pole
const trackHeight = 400;
const cart = new PointMass(400, trackHeight, 10, false);
const pole1 = new PointMass(400, 300, 2, false);
const pole2 = new PointMass(400, 200, 2, false);

// Constraints
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

const motor = new Actuator(cart, 2000);

// Brain: Higher complexity for double pendulum
const agent = new QLearningAgent(2);

let episode = 1;
let score = 0;
let maxScore = 0;
const scoreHistory: number[] = [];
const movingAverageHistory: number[] = [];
let currentMovingAvg = 0;
let currentStateStr = "";

function binVariable(value: number, min: number, max: number, bins: number): number {
    let clamped = Math.max(min, Math.min(max, value));
    let normalized = (clamped - min) / (max - min);
    let bin = Math.floor(normalized * bins);
    return Math.min(bin, bins - 1);
}

// Discretization for double pendulum needs more resolution
function discretize(cartX: number, pole1Angle: number, pole2Angle: number): string {
    const binX = binVariable(cartX, 50, 750, 4);
    const binA1 = binVariable(pole1Angle, -1, 1, 8);
    const binA2 = binVariable(pole2Angle, -1, 1, 8);
    return `${binX}-${binA1}-${binA2}`;
}

function resetWorld() {
    cart.position.x = 400;
    cart.position.y = trackHeight;
    cart.oldPosition.x = 400;
    cart.oldPosition.y = trackHeight;
    pole1.position.x = 400; pole1.position.y = 300;
    pole2.position.x = 400; pole2.position.y = 200;
    pole1.oldPosition.x = 400; pole1.oldPosition.y = 300;
    pole2.oldPosition.x = 400; pole2.oldPosition.y = 200;
    motor.activeDirection = 0;
}

function senseUniverse() {
    const dx1 = pole1.position.x - cart.position.x;
    const dy1 = cart.position.y - pole1.position.y;
    const dx2 = pole2.position.x - pole1.position.x;
    const dy2 = pole1.position.y - pole2.position.y;
    return {
        cartX: cart.position.x,
        a1: Math.atan2(dx1, dy1),
        a2: Math.atan2(dx2, dy2)
    };
}

const FIXED_DT = 0.016;
resetWorld();
let s = senseUniverse();
currentStateStr = discretize(s.cartX, s.a1, s.a2);

function step() {
    const stepsPerFrame = parseInt(stepsInput.value) || 1;

    for (let i = 0; i < stepsPerFrame; i++) {
        const action = agent.getAction(currentStateStr);
        motor.activeDirection = action === 0 ? -1 : 1;
        motor.apply();
        env.update(FIXED_DT);

        const next = senseUniverse();
        const isDead = Math.abs(next.a1) > 0.8 || Math.abs(next.a2) > 0.8 || next.cartX < 50 || next.cartX > 750;
        const reward = isDead ? -100 : 1;
        const nextStr = discretize(next.cartX, next.a1, next.a2);

        agent.learn(currentStateStr, action, reward, nextStr, isDead);

        if (isDead) {
            if (score > maxScore) maxScore = score;
            scoreHistory.push(score);
            if (scoreHistory.length > 50) scoreHistory.shift();
            currentMovingAvg = scoreHistory.reduce((a, b) => a + b, 0) / scoreHistory.length;
            movingAverageHistory.push(currentMovingAvg);
            if (movingAverageHistory.length > 100) movingAverageHistory.shift();
            (agent as any).epsilon *= 0.9999;
            resetWorld();
            episode++;
            score = 0;
            currentStateStr = discretize(400, 0, 0);
        } else {
            score += reward;
            currentStateStr = nextStr;
        }
    }

    renderer.render(env);
    ctx.fillStyle = 'white';
    ctx.fillText(`Episode: ${episode} | Score: ${score} | MovAvg: ${Math.round(currentMovingAvg)}`, 20, 40);
    requestAnimationFrame(step);
}

requestAnimationFrame(step);