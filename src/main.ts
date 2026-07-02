import { Environment } from './engine/Environment';
import { CanvasRenderer } from './renderer/CanvasRenderer';
import { PointMass } from './state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from './engine/Constraint'; // Import the constraint
import { Actuator } from './engine/Actuator';

// 1. Initialize Canvas and Renderer
const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
canvas.width = 800;
canvas.height = 600;
const renderer = new CanvasRenderer(canvas);

// 2. Initialize Physics Environment
const env = new Environment(9.81);


// 3. Populate our world: The Passive Cart-Pole
const trackHeight = 400;

// The Cart (Heavy base, unpinned so it can slide)
const cart = new PointMass(400, trackHeight, 10, false);

// The Pole Mass (Lighter, placed above and slightly to the right to simulate falling)
const poleTop = new PointMass(450, 200, 2, false);

// Connect the Cart to the Pole (The structural bone)
const bone = new DistanceConstraint(cart, poleTop, 200);

// Lock the Cart to the Y-axis (The steel track)
const track = new AxisConstraint(cart, trackHeight);


// Keep the Cart on the screen! (0 bounce so it hits the edge and stops abruptly like a real track)
const screenBounds = new BoundaryConstraint(cart, canvas.width, canvas.height, 0);

env.addPoint(cart);
env.addPoint(poleTop);
env.addConstraint(bone);
env.addConstraint(track);
env.addConstraint(screenBounds);

// --- The Motor ---
const motor = new Actuator(cart, 4000); // 4000 is the thrust strength. Tune this!

// Listen to the Human (or later, the AI)
window.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowLeft') motor.activeDirection = -1;
    if (e.code === 'ArrowRight') motor.activeDirection = 1;
});

window.addEventListener('keyup', (e) => {
    if (e.code === 'ArrowLeft' && motor.activeDirection === -1) motor.activeDirection = 0;
    if (e.code === 'ArrowRight' && motor.activeDirection === 1) motor.activeDirection = 0;
});

// 4. The Main Loop
let lastTime = performance.now();

function step(currentTime: number) {
    const dt = (currentTime - lastTime) / 1000;
    lastTime = currentTime;

    // 1. Fire the motor
    motor.apply();

    // 2. Run the math
    env.update(dt);

    // 3. Draw the screen
    renderer.render(env);

    requestAnimationFrame(step);
}

requestAnimationFrame(step);