import { PointMass } from '../state/PointMass';
import { Vector } from './Vector';

export class Actuator {
    public target: PointMass;
    public thrustPower: number;
    public activeDirection: number = 0; // -1 for Left, 1 for Right, 0 for off

    constructor(target: PointMass, thrustPower: number = 3000) {
        this.target = target;
        this.thrustPower = thrustPower;
    }

    // This must be called every frame right before the physics update
    apply() {
        if (this.activeDirection !== 0) {
            // Generate a purely horizontal force vector
            const force = new Vector(this.activeDirection * this.thrustPower, 0);
            this.target.applyForce(force);
        }
    }
}