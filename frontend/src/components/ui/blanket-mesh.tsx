import { useEffect, useRef } from "react";
import { cn } from "../../lib/utils";

interface BlanketMeshProps {
  className?: string;
  gridGap?: number;
  isFlareActive?: boolean;
}

interface Point {
  x: number;
  y: number;
  originX: number;
  originY: number;
  vx: number;
  vy: number;
}

export const BlanketMesh = ({
  className,
  gridGap = 36,
  isFlareActive = false,
}: BlanketMeshProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    let points: Point[][] = [];
    let cols = Math.ceil(width / gridGap) + 1;
    let rows = Math.ceil(height / gridGap) + 1;

    const initGrid = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      cols = Math.ceil(width / gridGap) + 1;
      rows = Math.ceil(height / gridGap) + 1;
      points = [];

      for (let r = 0; r < rows; r++) {
        const row: Point[] = [];
        for (let c = 0; c < cols; c++) {
          const x = c * gridGap;
          const y = r * gridGap;
          row.push({
            x,
            y,
            originX: x,
            originY: y,
            vx: 0,
            vy: 0,
          });
        }
        points.push(row);
      }
    };

    initGrid();

    // Mouse tracking for blanket drag
    let mouse = { x: -1000, y: -1000, prevX: -1000, prevY: -1000, speedX: 0, speedY: 0 };
    let timeoutId: number;

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const currentX = e.clientX - rect.left;
      const currentY = e.clientY - rect.top;

      mouse.speedX = currentX - (mouse.prevX === -1000 ? currentX : mouse.prevX);
      mouse.speedY = currentY - (mouse.prevY === -1000 ? currentY : mouse.prevY);
      mouse.prevX = mouse.x = currentX;
      mouse.prevY = mouse.y = currentY;

      window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(() => {
        mouse.speedX = 0;
        mouse.speedY = 0;
      }, 50);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("resize", initGrid);

    const spring = 0.045;
    const damping = 0.88;
    const influenceRadius = 160;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // 1. Update Physics
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const p = points[r][c];

          // Cursor displacement (elastic blanket effect)
          const dx = mouse.x - p.x;
          const dy = mouse.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < influenceRadius && dist > 0) {
            const force = (1 - dist / influenceRadius);
            // Move along cursor motion + push slightly outward
            const pushFactor = isFlareActive ? 18 : 12;
            p.vx += (mouse.speedX * 0.4 + (dx / dist) * -1.5) * force * pushFactor * 0.05;
            p.vy += (mouse.speedY * 0.4 + (dy / dist) * -1.5) * force * pushFactor * 0.05;
          }

          // Spring force back to rest position
          const forceX = (p.originX - p.x) * spring;
          const forceY = (p.originY - p.y) * spring;

          p.vx += forceX;
          p.vy += forceY;

          // Damping
          p.vx *= damping;
          p.vy *= damping;

          p.x += p.vx;
          p.y += p.vy;
        }
      }

      // 2. Draw Horizontal Mesh Lines
      const baseLineColor = isFlareActive
        ? "rgba(255, 51, 75, 0.12)"
        : "rgba(0, 229, 255, 0.12)";

      ctx.lineWidth = 1;

      for (let r = 0; r < rows; r++) {
        ctx.beginPath();
        ctx.moveTo(points[r][0].x, points[r][0].y);
        for (let c = 1; c < cols; c++) {
          const p = points[r][c];
          ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = baseLineColor;
        ctx.stroke();
      }

      // 3. Draw Vertical Mesh Lines & Glowing Nodes on active displacement
      for (let c = 0; c < cols; c++) {
        ctx.beginPath();
        ctx.moveTo(points[0][c].x, points[0][c].y);
        for (let r = 1; r < rows; r++) {
          const p = points[r][c];
          ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = baseLineColor;
        ctx.stroke();
      }

      // 4. Draw glowing nodes when displaced
      for (let r = 0; r < rows; r += 2) {
        for (let c = 0; c < cols; c += 2) {
          const p = points[r][c];
          const disp = Math.hypot(p.x - p.originX, p.y - p.originY);

          if (disp > 1.5) {
            const glowAlpha = Math.min(0.9, disp / 12);
            ctx.beginPath();
            ctx.arc(p.x, p.y, Math.min(3.5, 1.2 + disp * 0.18), 0, Math.PI * 2);
            ctx.fillStyle = isFlareActive
              ? `rgba(255, 82, 82, ${glowAlpha})`
              : `rgba(0, 229, 255, ${glowAlpha})`;
            ctx.shadowBlur = disp * 1.5;
            ctx.shadowColor = isFlareActive ? "#ff334b" : "#00e5ff";
            ctx.fill();
            ctx.shadowBlur = 0;
          }
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("resize", initGrid);
    };
  }, [gridGap, isFlareActive]);

  return (
    <canvas
      ref={canvasRef}
      className={cn(
        "fixed inset-0 pointer-events-none z-0 h-full w-full opacity-70 transition-opacity duration-700",
        className
      )}
    />
  );
};