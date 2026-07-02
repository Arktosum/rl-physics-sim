export class Vector {
    // The 'public' keyword in the constructor automatically creates and assigns this.x and this.y
    constructor(public x: number, public y: number) { }

    // Addition: A + B
    add(v: Vector): Vector {
        return new Vector(this.x + v.x, this.y + v.y);
    }

    // Subtraction: A - B (Creates a vector pointing from v to this)
    sub(v: Vector): Vector {
        return new Vector(this.x - v.x, this.y - v.y);
    }

    // Scalar Multiplication: Scales the vector by a number
    mult(s: number): Vector {
        return new Vector(this.x * s, this.y * s);
    }

    // Magnitude: The physical length of the vector using Pythagorean theorem
    mag(): number {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    // Normalization: Keeps the direction, but forces the length to be exactly 1
    normalize(): Vector {
        const m = this.mag();
        if (m === 0) return new Vector(0, 0); // Prevent divide-by-zero errors
        return new Vector(this.x / m, this.y / m);
    }
}