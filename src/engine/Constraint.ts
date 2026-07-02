import { PointMass } from '../state/PointMass';




// The universal contract for all structural rules
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

        // If no length is provided, just lock in whatever distance they are currently at
        if (length === undefined) {
            this.targetLength = p1.position.sub(p2.position).mag();
        } else {
            this.targetLength = length;
        }
    }

    // The violent correction step
    resolve() {
        // 1. Find the axis and distance
        const delta = this.p2.position.sub(this.p1.position);
        const currentDistance = delta.mag();

        // Prevent division by zero if they occupy the exact same pixel
        if (currentDistance === 0) return;

        // 2. Calculate the error (how far off we are from the target length)
        const error = currentDistance - this.targetLength;

        // 3. Find the normalized direction vector pointing from p1 to p2
        const direction = delta.normalize();

        // 4. Calculate Inverse Mass (w = 1 / mass)
        // If an object is pinned, its inverse mass is effectively 0 (it won't move)
        const w1 = this.p1.isPinned ? 0 : 1 / this.p1.mass;
        const w2 = this.p2.isPinned ? 0 : 1 / this.p2.mass;
        const wTotal = w1 + w2;

        // If both are pinned, do nothing
        if (wTotal === 0) return;

        // 5. Distribute the correction based on mass
        // Lighter objects move more. Pinned objects move 0.
        const correctionMagnitude = error / wTotal;

        if (!this.p1.isPinned) {
            // Push p1 forward along the line
            const correction1 = direction.mult(correctionMagnitude * w1);
            this.p1.position = this.p1.position.add(correction1);
        }

        if (!this.p2.isPinned) {
            // Push p2 backward along the line
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

        // Violently teleport the Y coordinate back to the track
        // The X coordinate is completely ignored, preserving horizontal inertia
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

        // Calculate the velocity we had before impact
        const velX = this.point.position.x - this.point.oldPosition.x;
        const velY = this.point.position.y - this.point.oldPosition.y;

        const radius = this.point.radius || 10; // Assuming a default visual radius of 10

        // 1. Left Wall
        if (this.point.position.x < radius) {
            this.point.position.x = radius;
            // Hack the past to push it right
            this.point.oldPosition.x = this.point.position.x + velX * this.bounce;
        }
        // 2. Right Wall
        else if (this.point.position.x > this.width - radius) {
            this.point.position.x = this.width - radius;
            // Hack the past to push it left
            this.point.oldPosition.x = this.point.position.x + velX * this.bounce;
        }

        // 3. Ceiling
        if (this.point.position.y < radius) {
            this.point.position.y = radius;
            this.point.oldPosition.y = this.point.position.y + velY * this.bounce;
        }
        // 4. Floor
        else if (this.point.position.y > this.height - radius) {
            this.point.position.y = this.height - radius;
            this.point.oldPosition.y = this.point.position.y + velY * this.bounce;
        }
    }
}
