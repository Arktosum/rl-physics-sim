import { Vector } from '../engine/Vector';

export class PointMass {
    public position: Vector;
    public oldPosition: Vector; // We need this for our Semi-Implicit Euler integration
    public acceleration: Vector;
    public mass: number;
    public radius: number;
    public isPinned: boolean; // If true, the object ignores physics (like a wall or a pivot)

    constructor(x: number, y: number, mass: number = 1, isPinned: boolean = false) {
        this.position = new Vector(x, y);
        this.oldPosition = new Vector(x, y);
        this.acceleration = new Vector(0, 0);
        this.mass = mass;
        this.radius = 10; // For rendering
        this.isPinned = isPinned;
    }

    // Apply a continuous force (like gravity)
    applyForce(force: Vector) {
        if (this.isPinned) return;
        // Newton's Second Law: a = F/m
        const a = force.mult(1 / this.mass);
        this.acceleration = this.acceleration.add(a);
    }
    // Steps the physics forward in time
    update(dt: number) {
        if (this.isPinned) return;

        // 1. Calculate current velocity (Position - Old Position)
        const velocity = this.position.sub(this.oldPosition);

        // 2. Save the current position before we change it
        this.oldPosition = this.position;

        // 3. Calculate new position: pos + vel + (acceleration * dt * dt)
        // We multiply acceleration by dt squared as per the Verlet integration formula
        const accelerationStep = this.acceleration.mult(dt * dt);
        this.position = this.position.add(velocity).add(accelerationStep);

        // 4. Reset acceleration for the next frame so forces don't stack infinitely
        this.acceleration = new Vector(0, 0);
    }
}