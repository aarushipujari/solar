import { useEffect, useRef } from "react";
import { cn } from "../../lib/utils";

interface VortexProps {
  children?: React.ReactNode;
  className?: string;
  containerClassName?: string;
  particleCount?: number;
  isFlareActive?: boolean;
}

export const Vortex = ({
  children,
  className,
  containerClassName,
  particleCount = 180,
  isFlareActive = false,
}: VortexProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    const particles: Array<{
      x: number;
      y: number;
      radius: number;
      angle: number;
      distance: number;
      speed: number;
      color: string;
      alpha: number;
    }> = [];

    const calmColors = ["#00e5ff", "#00b0ff", "#4fc3f7", "#7c4dff", "#00e676"];
    const flareColors = ["#ff334b", "#ff5252", "#ffb300", "#ff9100", "#ff1744"];

    for (let i = 0; i < particleCount; i++) {
      const colors = isFlareActive ? flareColors : calmColors;
      particles.push({
        x: width / 2,
        y: height / 2,
        radius: Math.random() * 2.2 + 0.8,
        angle: Math.random() * Math.PI * 2,
        distance: Math.random() * (Math.max(width, height) / 2),
        speed: (Math.random() * 0.008 + 0.003) * (isFlareActive ? 2.5 : 1),
        color: colors[Math.floor(Math.random() * colors.length)],
        alpha: Math.random() * 0.7 + 0.3,
      });
    }

    const render = () => {
      // Create motion blur trail
      ctx.fillStyle = "rgba(3, 7, 18, 0.25)";
      ctx.fillRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;

      particles.forEach((p) => {
        p.angle += p.speed;
        p.distance += isFlareActive ? 1.2 : 0.4;

        if (p.distance > Math.max(width, height) / 1.5) {
          p.distance = Math.random() * 40;
          p.angle = Math.random() * Math.PI * 2;
        }

        const px = centerX + Math.cos(p.angle) * p.distance;
        const py = centerY + Math.sin(p.angle) * p.distance * 0.6; // Elliptical orbit

        ctx.beginPath();
        ctx.arc(px, py, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.shadowBlur = isFlareActive ? 12 : 6;
        ctx.shadowColor = p.color;
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [particleCount, isFlareActive]);

  return (
    <div className={cn("relative h-full w-full overflow-hidden", containerClassName)}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-0 h-full w-full pointer-events-none"
      />
      <div className={cn("relative z-10", className)}>{children}</div>
    </div>
  );
};