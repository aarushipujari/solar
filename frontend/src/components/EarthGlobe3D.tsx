import { useEffect, useRef } from "react";
import * as THREE from "three";

interface EarthGlobe3DProps {
  isCritical?: boolean;
  className?: string;
}

export const EarthGlobe3D = ({
  isCritical = false,
  className = "",
}: EarthGlobe3DProps) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 400;
    const height = currentMount.clientHeight || 400;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 7;
    camera.position.y = 1.5;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    // Group for rotation
    const globeGroup = new THREE.Group();
    scene.add(globeGroup);

    // 1. Earth Core Sphere (Dark Blue / Cyber Grid)
    const earthGeo = new THREE.SphereGeometry(2, 36, 36);
    const earthMat = new THREE.MeshBasicMaterial({
      color: 0x071530,
      wireframe: true,
      transparent: true,
      opacity: 0.35,
    });
    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    globeGroup.add(earthMesh);

    // Solid inner core
    const innerGeo = new THREE.SphereGeometry(1.96, 32, 32);
    const innerMat = new THREE.MeshBasicMaterial({
      color: 0x030a1c,
    });
    const innerMesh = new THREE.Mesh(innerGeo, innerMat);
    globeGroup.add(innerMesh);

    // 2. India Location Highlight Pin (Lat ~20°N, Lon ~78°E)
    const phi = (90 - 20) * (Math.PI / 180);
    const theta = (78 + 180) * (Math.PI / 180);
    const pinX = -(2.05 * Math.sin(phi) * Math.cos(theta));
    const pinZ = 2.05 * Math.sin(phi) * Math.sin(theta);
    const pinY = 2.05 * Math.cos(phi);

    const pinGeo = new THREE.SphereGeometry(0.08, 16, 16);
    const pinMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const pinMesh = new THREE.Mesh(pinGeo, pinMat);
    pinMesh.position.set(pinX, pinY, pinZ);
    globeGroup.add(pinMesh);

    // 3. Magnetosphere Shield Rings
    const shieldColor = isCritical ? 0xff334b : 0x00e5ff;
    const shieldMat = new THREE.MeshBasicMaterial({
      color: shieldColor,
      transparent: true,
      opacity: isCritical ? 0.6 : 0.25,
      wireframe: true,
    });

    const shieldRings: THREE.Mesh[] = [];
    for (let i = 0; i < 4; i++) {
      const ringGeo = new THREE.TorusGeometry(2.6 + i * 0.4, 0.02, 16, 64);
      const ring = new THREE.Mesh(ringGeo, shieldMat);
      ring.rotation.x = Math.PI / 2 + (i - 1.5) * 0.2;
      ring.rotation.y = (i - 1.5) * 0.15;
      globeGroup.add(ring);
      shieldRings.push(ring);
    }

    // 4. Satellite Orbits (NavIC GSO & Gaganyaan LEO)
    const navicOrbitGeo = new THREE.BufferGeometry();
    const navicPoints: THREE.Vector3[] = [];
    const orbitRadius = 3.6;
    for (let i = 0; i <= 64; i++) {
      const angle = (i / 64) * Math.PI * 2;
      navicPoints.push(
        new THREE.Vector3(
          Math.cos(angle) * orbitRadius,
          Math.sin(angle) * (orbitRadius * 0.35),
          Math.sin(angle) * orbitRadius
        )
      );
    }
    navicOrbitGeo.setFromPoints(navicPoints);
    const navicOrbitMat = new THREE.LineBasicMaterial({
      color: 0xffb300,
      transparent: true,
      opacity: 0.4,
    });
    const navicOrbitLine = new THREE.Line(navicOrbitGeo, navicOrbitMat);
    globeGroup.add(navicOrbitLine);

    // Satellite Markers
    const satGeo = new THREE.BoxGeometry(0.12, 0.12, 0.12);
    const satMat = new THREE.MeshBasicMaterial({ color: 0xffb300 });
    const satMesh = new THREE.Mesh(satGeo, satMat);
    globeGroup.add(satMesh);

    // LEO Gaganyaan Satellite
    const leoSatGeo = new THREE.SphereGeometry(0.06, 12, 12);
    const leoSatMat = new THREE.MeshBasicMaterial({ color: 0x00e676 });
    const leoSatMesh = new THREE.Mesh(leoSatGeo, leoSatMat);
    globeGroup.add(leoSatMesh);

    // Mouse Interaction
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;

      globeGroup.rotation.y += deltaX * 0.006;
      globeGroup.rotation.x += deltaY * 0.006;

      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    currentMount.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    // Animation Loop
    let satAngle = 0;
    let animId: number;

    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (!isDragging) {
        globeGroup.rotation.y += 0.003;
      }

      satAngle += 0.015;
      satMesh.position.x = Math.cos(satAngle) * orbitRadius;
      satMesh.position.y = Math.sin(satAngle) * (orbitRadius * 0.35);
      satMesh.position.z = Math.sin(satAngle) * orbitRadius;

      // LEO Orbit (Fast)
      leoSatMesh.position.x = Math.cos(satAngle * 2.5) * 2.3;
      leoSatMesh.position.y = Math.sin(satAngle * 2.5) * 2.3;
      leoSatMesh.position.z = Math.sin(satAngle * 2.5) * 0.5;

      // Pulse Magnetosphere shield during critical alerts
      if (isCritical) {
        const pulse = 1 + Math.sin(Date.now() * 0.008) * 0.06;
        shieldRings.forEach((r) => r.scale.set(pulse, pulse, pulse));
      }

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!currentMount) return;
      const w = currentMount.clientWidth;
      const h = currentMount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      currentMount.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [isCritical]);

  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      <div ref={mountRef} className="w-full h-[320px] md:h-[420px] cursor-grab active:cursor-grabbing" />
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-3 px-3 py-1 rounded-full bg-space-900/90 border border-white/10 text-[10px] font-mono text-slate-400 backdrop-blur-md">
        <span className="flex items-center gap-1 text-amber-400">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400" /> NavIC GSO
        </span>
        <span className="flex items-center gap-1 text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Gaganyaan LEO
        </span>
        <span className="flex items-center gap-1 text-cyan-400">
          <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> ISRO Earth Hub
        </span>
      </div>
    </div>
  );
};