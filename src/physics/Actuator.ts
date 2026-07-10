import { PointMass } from '../state/PointMass';
import { Vector } from './Vector';

export class Actuator {
    public target: PointMass;
    public thrustPower: number;
    public activeDirection: number = 0; // -1 left, 1 right, 0 off

    constructor(target: PointMass, thrustPower: number = 3000) {
        this.target = target;
        this.thrustPower = thrustPower;
    }

    // call before env.update() each frame, not after
    apply() {
        if (this.activeDirection !== 0) {
            const force = new Vector(this.activeDirection * this.thrustPower, 0);
            this.target.applyForce(force);
        }
    }
}