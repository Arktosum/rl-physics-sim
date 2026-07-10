import { PointMass } from '../state/PointMass';

export interface Constraint {
    resolve(): void;
}

export class DistanceConstraint implements Constraint {
    public p1: PointMass;
    public p2: PointMass;
    private targetLength: number;

    constructor(p1: PointMass, p2: PointMass, length?: number) {
        this.p1 = p1;
        this.p2 = p2;

        // no length given -> lock whatever distance they start at
        if (length === undefined) {
            this.targetLength = p1.position.sub(p2.position).mag();
        } else {
            this.targetLength = length;
        }
    }

    resolve() {
        const delta = this.p2.position.sub(this.p1.position);
        const currentDistance = delta.mag();

        if (currentDistance === 0) return; // coincident points, direction undefined

        const error = currentDistance - this.targetLength;
        const direction = delta.normalize();

        // inverse mass weighting: pinned points have w=0 so all correction goes to the other end
        const w1 = this.p1.isPinned ? 0 : 1 / this.p1.mass;
        const w2 = this.p2.isPinned ? 0 : 1 / this.p2.mass;
        const wTotal = w1 + w2;

        if (wTotal === 0) return; // both pinned

        const correctionMagnitude = error / wTotal;

        if (!this.p1.isPinned) {
            const correction1 = direction.mult(correctionMagnitude * w1);
            this.p1.position = this.p1.position.add(correction1);
        }

        if (!this.p2.isPinned) {
            const correction2 = direction.mult(-correctionMagnitude * w2);
            this.p2.position = this.p2.position.add(correction2);
        }
    }
}

export class AxisConstraint implements Constraint {
    public point: PointMass;
    public lockedY: number;

    constructor(point: PointMass, lockedY: number) {
        this.point = point;
        this.lockedY = lockedY;
    }

    resolve() {
        if (this.point.isPinned) return;

        // only clamps y; x is left alone so horizontal inertia carries through
        this.point.position.y = this.lockedY;
    }
}

export class BoundaryConstraint implements Constraint {
    public point: PointMass;
    public width: number;
    public height: number;
    public bounce: number;

    constructor(point: PointMass, width: number, height: number, bounce: number = 0.3) {
        this.point = point;
        this.width = width;
        this.height = height;
        this.bounce = bounce;
    }

    resolve() {
        if (this.point.isPinned) return;

        const velX = this.point.position.x - this.point.oldPosition.x;
        const velY = this.point.position.y - this.point.oldPosition.y;

        const radius = this.point.radius || 10;

        // Verlet has no explicit velocity, so bounce is done by setting oldPosition
        // on the opposite side of position - that's what the next update() reads as velocity
        if (this.point.position.x < radius) {
            this.point.position.x = radius;
            this.point.oldPosition.x = this.point.position.x + velX * this.bounce;
        }
        else if (this.point.position.x > this.width - radius) {
            this.point.position.x = this.width - radius;
            this.point.oldPosition.x = this.point.position.x + velX * this.bounce;
        }

        if (this.point.position.y < radius) {
            this.point.position.y = radius;
            this.point.oldPosition.y = this.point.position.y + velY * this.bounce;
        }
        else if (this.point.position.y > this.height - radius) {
            this.point.position.y = this.height - radius;
            this.point.oldPosition.y = this.point.position.y + velY * this.bounce;
        }
    }
}
