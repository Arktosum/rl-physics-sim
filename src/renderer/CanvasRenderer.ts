import { Environment } from '../physics/Environment';

// subset of Environment's shape actually read here. Environment satisfies this
// structurally already; exists so a plain data snapshot posted from a training
// worker (points + constraints, no class instances/methods) can be rendered too -
// structured-clone can't preserve prototypes across postMessage anyway.
export interface RenderablePoint {
    position: { x: number; y: number };
    mass: number;
    isPinned: boolean;
}
export interface RenderableConstraint {
    p1?: { position: { x: number; y: number } };
    p2?: { position: { x: number; y: number } };
    lockedY?: number;
}
export interface RenderableEnvironment {
    points: RenderablePoint[];
    constraints: RenderableConstraint[];
}

export class CanvasRenderer {
    private ctx: CanvasRenderingContext2D;
    private canvas: HTMLCanvasElement;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
    }

    render(env: RenderableEnvironment | Environment) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // constraints drawn before points so joints render on top of the lines
        this.ctx.strokeStyle = '#555577';
        this.ctx.lineWidth = 2;

        for (const constraint of env.constraints) {
            if ('p1' in constraint && 'p2' in constraint) {
                const c = constraint as any;
                this.ctx.strokeStyle = '#555577';
                this.ctx.beginPath();
                this.ctx.moveTo(c.p1.position.x, c.p1.position.y);
                this.ctx.lineTo(c.p2.position.x, c.p2.position.y);
                this.ctx.stroke();
            }
            else if ('lockedY' in constraint) {
                // axis track (cart rail) - horizontal line at the locked y
                const c = constraint as any;
                this.ctx.strokeStyle = '#334433';
                this.ctx.beginPath();
                this.ctx.moveTo(0, c.lockedY);
                this.ctx.lineTo(this.canvas.width, c.lockedY);
                this.ctx.stroke();
            }
        }

        for (const point of env.points) {
            this.ctx.beginPath();
            this.ctx.fillStyle = point.isPinned ? '#38bdf8' : '#ff0055';
            const visualRadius = Math.max(5, Math.min(30, point.mass * 5));
            this.ctx.arc(point.position.x, point.position.y, visualRadius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
}