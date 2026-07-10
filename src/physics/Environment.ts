import { Vector } from './Vector';
import { PointMass } from '../state/PointMass';
import { type Constraint } from './Constraint';

export class Environment {
    public points: PointMass[] = [];
    public constraints: Constraint[] = [];
    public gravity: Vector;

    constructor(gravityY: number = 9.8) {
        // positive y is down in screen coords, so gravity points +y
        this.gravity = new Vector(0, gravityY * 100); // x100 so pixel units read as roughly meters
    }

    addPoint(point: PointMass) {
        this.points.push(point);
    }

    addConstraint(constraint: Constraint) {
        this.constraints.push(constraint);
    }

    update(dt: number) {
        if (dt > 0.1) dt = 0.1; // clamp tab-switch/stall spikes so integration doesn't blow up

        for (const point of this.points) {
            // scale by mass so applyForce's F/m division cancels out and all bodies fall at the same rate
            const gravityForce = this.gravity.mult(point.mass);
            point.applyForce(gravityForce);
        }

        for (const point of this.points) {
            point.update(dt);
        }

        // Gauss-Seidel style relaxation: constraints fight each other, so resolve
        // repeatedly until they converge instead of just once
        const iterations = 50;
        for (let i = 0; i < iterations; i++) {
            for (const constraint of this.constraints) {
                constraint.resolve();
            }
        }
    }
}