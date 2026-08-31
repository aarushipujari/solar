import React from "react";
import { motion } from "framer-motion";
import { Vortex } from "./ui/vortex";
import { EarthGlobe3D } from "./EarthGlobe3D";
import { Sun, ArrowRight, ShieldCheck, Activity } from "lucide-react";

interface HeroLandingProps {
  onEnterDashboard: () => void;
  isFlareActive: boolean;
  activeRegion: string;
}

export const HeroLanding: React.FC<HeroLandingProps> = ({
  onEnterDashboard,
  isFlareActive,
  activeRegion,
}) => {
  return (
    <div className="relative min-h-[90vh] flex flex-col justify-center items-center overflow-hidden py-12 px-4">
      {/* Vortex Particle Canvas */}
      <Vortex
        isFlareActive={isFlareActive}
        className="flex flex-col items-center justify-center w-full max-w-7xl mx-auto"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center w-full">
          {/* Left Column: Mission Hook & CTA */}
          <div className="lg:col-span-7 space-y-6 text-left">
            {/* Mission Badge */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-space-900/90 border border-cyan-500/30 backdrop-blur-xl shadow-lg shadow-cyan-500/10 text-xs font-mono text-cyan-300"
            >
              <Sun className="h-4 w-4 text-orange-400 animate-spin-slow" />
              <span>ISRO ADITYA-L1 SUIT PAYLOAD (279.6 nm)</span>
              <span className="text-white/30">•</span>
              <span className="text-orange-400 font-bold">SIH 2026 GRAND FINALE</span>
            </motion.div>

            {/* Main Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-3xl md:text-5xl lg:text-6xl font-extrabold text-white tracking-tight leading-[1.15]"
            >
              Predicting Solar Eruptions{" "}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">
                24 to 48 Hours
              </span>{" "}
              Before Earth Impact.
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-sm md:text-base text-slate-300 leading-relaxed max-w-2xl font-sans"
            >
              Transforming reactive space mitigation into proactive national asset defense.
              Powered by a 4-channel spatio-temporal ConvLSTM neural architecture with authentic
              PyTorch autograd Grad-CAM explainability and 12-Fold LORO-CV validation.
            </motion.p>

            {/* Glowing Action Button (The "Hook" CTA) */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="pt-2 flex flex-wrap items-center gap-4"
            >
              <button
                onClick={onEnterDashboard}
                className="relative group/btn overflow-hidden rounded-2xl p-[2px] cursor-pointer transition-transform hover:scale-105 active:scale-95 shadow-2xl shadow-cyan-500/25"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 animate-shimmer" />
                <div className="relative px-8 py-4 bg-space-950 rounded-2xl flex items-center gap-3 text-sm md:text-base font-bold text-white transition-colors group-hover/btn:bg-space-900">
                  <span>LAUNCH MISSION DASHBOARD</span>
                  <ArrowRight className="h-5 w-5 text-cyan-400 transition-transform group-hover/btn:translate-x-1" />
                </div>
              </button>

              <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-space-900/80 px-4 py-3 rounded-2xl border border-white/10">
                <Activity className="h-4 w-4 text-emerald-400 animate-pulse" />
                <span>Active Target: <strong className="text-cyan-300">{activeRegion}</strong></span>
              </div>
            </motion.div>

            {/* Quick Metrics Strip */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="grid grid-cols-3 gap-3 pt-4 border-t border-white/10 font-mono text-xs"
            >
              <div className="bg-space-950/70 p-3 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">SCIENTIFIC RIGOR</span>
                <span className="text-cyan-400 font-bold">12-Fold LORO-CV</span>
              </div>
              <div className="bg-space-950/70 p-3 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">PRECURSOR WINDOW</span>
                <span className="text-amber-400 font-bold">24-48 Hours</span>
              </div>
              <div className="bg-space-950/70 p-3 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">EXPLAINABLE AI</span>
                <span className="text-emerald-400 font-bold">PyTorch Autograd</span>
              </div>
            </motion.div>
          </div>

          {/* Right Column: 3D Earth Globe & Shield Simulation */}
          <div className="lg:col-span-5 relative flex flex-col items-center justify-center">
            <div className="relative w-full rounded-3xl border border-white/10 bg-space-900/40 p-4 backdrop-blur-2xl shadow-2xl">
              <div className="flex items-center justify-between px-2 mb-2 font-mono text-xs text-slate-400">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className={`h-4 w-4 ${isFlareActive ? "text-rose-400" : "text-cyan-400"}`} />
                  3D Magnetosphere Shield & Orbits
                </span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  isFlareActive ? "bg-rose-500/20 text-rose-300" : "bg-cyan-500/20 text-cyan-300"
                }`}>
                  {isFlareActive ? "SHIELD COMPRESSED" : "GEO SHIELD NOMINAL"}
                </span>
              </div>

              {/* Three.js Interactive Earth Component */}
              <EarthGlobe3D isCritical={isFlareActive} />

              <div className="mt-2 text-center text-[11px] text-slate-400 font-mono">
                Interactive: Click & Drag to inspect NavIC GSO and Gaganyaan LEO orbits.
              </div>
            </div>
          </div>
        </div>
      </Vortex>
    </div>
  );
};