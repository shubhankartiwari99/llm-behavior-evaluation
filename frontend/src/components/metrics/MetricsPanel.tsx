import React from "react";
import { BarChart3, Zap, BrainCircuit, Lightbulb } from "lucide-react";

export default function MetricsPanel({ metrics, interventionType, metadata }: any) {
  if (!metrics) return null;

  const collapseLabel = metrics.collapse_ratio < 0.3 ? "Heavy shaping" : 
                        metrics.collapse_ratio < 0.6 ? "Moderate shaping" : "Light shaping";
  
  const collapseColor = metrics.collapse_ratio < 0.3 ? "text-red-400" :
                        metrics.collapse_ratio < 0.6 ? "text-amber-400" : "text-emerald-400";

  const klValue = metrics.kl_divergence || 0.0;
  const klInterpretation = klValue < 0.1 ? "Minimal Shift" :
                           klValue <= 0.5 ? "Moderate Transformation" : "Strong Behavioral Shift";

  let insightText = "";
  if (metrics.collapse_ratio < 0.3 && metrics.kl_divergence > 1.0) {
      insightText = "Severe entropy collapse and high KL metric indicates structural transformation of the distribution manifold.";
  } else if (metrics.collapse_ratio > 0.6 && metrics.kl_divergence < 0.5) {
      insightText = "Minimal shaping. The distribution preserves raw topology without significant semantic shift.";
  } else {
      insightText = "Moderate shaping. Target policies compressed entropy while managing distributional shift.";
  }

  const formatSource = (src: string) => {
      if (!src) return "Unknown";
      if (src === "mock_inference_pipeline") return "Mock Pipeline";
      if (src === "kaggle_qwen7b_pipeline") return "Kaggle (Qwen 7B)";
      if (src === "gguf_inference_pipeline_fallback") return "Local GGUF (Fallback)";
      if (src === "gguf_inference_pipeline") return "Local GGUF";
      return src;
  };

  return (
    <div className="space-y-4 p-5 border rounded-xl bg-slate-900/40 border-slate-800 shadow-2xl">
      <div className="flex items-center justify-between mb-2">
         <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-slate-400" />
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">System Trace Metrics</h3>
         </div>
         {metadata && (
            <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 uppercase tracking-wider">
               <span>Mode: <span className="text-amber-400">{formatSource(metadata.source)}</span></span>
               <span>Latency: <span className="text-cyan-400">{metadata.latency_ms}ms</span></span>
            </div>
         )}
      </div>
      
      {interventionType && (
        <div className="p-3 bg-cyan-950/30 border border-cyan-900/50 rounded-lg flex items-center mb-4">
          <Zap className="w-4 h-4 text-cyan-400 mr-2" />
          <span className="text-cyan-500 text-[10px] font-bold tracking-widest uppercase mr-3">Runtime Action</span> 
          <span className="font-mono text-cyan-100 text-xs px-2.5 py-1 bg-cyan-900/50 rounded-md border border-cyan-800/50 shadow-inner">{interventionType}</span>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800/80 hover:border-slate-700 transition-colors">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2 flex items-center gap-1.5">
             <BrainCircuit className="w-3 h-3" /> Collapse
          </div>
          <div className="flex items-baseline space-x-2">
            <span className={`text-3xl font-light tracking-tight ${collapseColor}`}>{metrics.collapse_ratio.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-[10px] font-medium text-slate-500 uppercase tracking-wider">{collapseLabel}</div>
        </div>

        <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800/80 hover:border-slate-700 transition-colors">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2 flex items-center gap-1.5">
            KL Divergence
          </div>
          <div className="flex items-baseline space-x-2">
             <span className="text-3xl font-light tracking-tight text-purple-400">{klValue.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-[10px] font-medium text-purple-500/70 uppercase tracking-wider">{klInterpretation}</div>
        </div>

        <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800/80 hover:border-slate-700 transition-colors">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2 flex items-center gap-1.5">
            Raw Entropy
          </div>
          <div className="flex items-baseline space-x-2">
             <span className="text-3xl font-light tracking-tight text-slate-200">{metrics.entropy_raw.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-[10px] font-medium text-slate-500 uppercase tracking-wider">Pre-Intervention</div>
        </div>

        <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800/80 hover:border-slate-700 transition-colors">
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2 flex items-center gap-1.5">
            Final Entropy
          </div>
          <div className="flex items-baseline space-x-2">
             <span className="text-3xl font-light tracking-tight text-slate-200">{metrics.entropy_final.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-[10px] font-medium text-slate-500 uppercase tracking-wider">Post-Intervention</div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-emerald-950/20 border border-emerald-900/30 rounded-lg flex items-start gap-3">
        <Lightbulb className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
        <div>
           <h4 className="text-[10px] text-emerald-500/80 font-bold uppercase tracking-widest mb-1.5">System Insight</h4>
           <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
             {insightText}
           </p>
        </div>
      </div>
    </div>
  );
}
