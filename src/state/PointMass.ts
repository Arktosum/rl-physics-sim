import { Vector } from '../physics/Vector';

export class PointMass {
    public position: Vector;
    public oldPosition: Vector; // Verlet integration derives velocity from this, not an explicit velocity field
    public acceleration: Vector;
    public mass: number;
    public radius: number;
    public isPinned: boolean; // pinned points are immovable (walls, pivots)

    constructor(x: number, y: number, mass: number = 1, isPinned: boolean = false) {
        this.position = new Vector(x, y);
        this.oldPosition = new Vector(x, y);
        this.acceleration = new Vector(0, 0);
        this.mass = mass;
        this.radius = 10;
        this.isPinned = isPinned;
    }

    applyForce(force: Vector) {
        if (this.isPinned) return;
        const a = force.mult(1 / this.mass);
        this.acceleration = this.acceleration.add(a);
    }

    update(dt: number) {
        if (this.isPinned) return;

        const velocity = this.position.sub(this.oldPosition);
        this.oldPosition = this.position;

        const accelerationStep = this.acceleration.mult(dt * dt);
        this.position = this.position.add(velocity).add(accelerationStep);

        // cleared each step - forces are re-applied fresh every frame (gravity, actuator, etc)
        this.acceleration = new Vector(0, 0);
    }
}