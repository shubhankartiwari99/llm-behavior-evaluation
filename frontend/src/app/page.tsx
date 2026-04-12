"use client"

import { useState } from "react"
import { useInference } from "@/hooks/useInference"
import { Activity, Play, Settings2, ShieldCheck, Cpu } from "lucide-react"
import OutputComparison from "@/components/outputs/OutputComparison"
import MetricsPanel from "@/components/metrics/MetricsPanel"

export default function Home() {
  const [prompt, setPrompt] = useState("")
  const [useMock, setUseMock] = useState(true)
  const { data, execute, loading } = useInference()

  return (
    <div className="min-h-screen bg-[#020617] text-slate-300 font-sans selection:bg-cyan-900/50">
      {/* Header */}
      <header className="border-b border-slate-800/60 bg-slate-900/30 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/20">
              <Activity className="w-5 h-5 text-cyan-400" />
            </div>
            <h1 className="text-sm font-bold tracking-widest text-slate-200 uppercase">
              Distribution Shaping <span className="text-cyan-500">Pipeline</span>
            </h1>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono text-slate-500">
            <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> Runtime Active</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Sidebar Controls */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-slate-400" />
                <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">Inference Parameters</h2>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between bg-slate-950 p-3 rounded-lg border border-slate-800/50">
                <span className="text-[10px] uppercase font-bold tracking-widest text-slate-400">Mode: <span className={useMock ? "text-cyan-400" : "text-amber-400"}>{useMock ? 'Mock (Fast)' : 'Real (Kaggle/Local)'}</span></span>
                <button
                  onClick={() => setUseMock(!useMock)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${useMock ? 'bg-cyan-500' : 'bg-slate-700'}`}
                >
                  <span className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${useMock ? 'translate-x-5' : 'translate-x-1'}`} />
                </button>
              </div>

              <div>
                <label className="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-2 block">Instruction Prompt</label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter a prompt to evaluate..."
                  className="w-full h-32 bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm text-slate-300 focus:outline-none focus:border-cyan-500/50 transition-colors resize-none font-serif"
                />
              </div>

              <button
                onClick={() => execute({ prompt, use_mock: useMock })}
                disabled={loading || !prompt.trim()}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 px-6 py-3.5 text-xs font-bold uppercase tracking-widest transition-all shadow-[0_0_20px_rgba(6,182,212,0.15)] disabled:shadow-none"
              >
                {loading ? <Cpu className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                {loading ? "Evaluating Pipeline..." : "Execute Evaluation"}
              </button>
            </div>
          </div>
          
          <div className="bg-slate-900/20 border border-slate-800/50 rounded-xl p-5">
             <div className="flex items-start gap-3">
               <ShieldCheck className="w-5 h-5 text-emerald-500 mt-0.5" />
               <div>
                  <h3 className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-1">System Boundary</h3>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    This evaluation interface explicitly isolates stochastic model generation from deterministic runtime interventions.
                  </p>
               </div>
             </div>
          </div>
        </div>

        {/* Main Display */}
        <div className="lg:col-span-8 space-y-6">
          {!data && !loading && (
             <div className="h-full min-h-[400px] border border-slate-800/50 border-dashed rounded-xl flex flex-col items-center justify-center text-slate-500 bg-slate-900/10">
               <Activity className="w-12 h-12 mb-4 opacity-20" />
               <p className="text-sm">Awaiting execution to visualize distribution shaping.</p>
             </div>
          )}

          {loading && (
            <div className="h-full min-h-[400px] border border-cyan-900/30 rounded-xl flex flex-col items-center justify-center text-cyan-500/50 bg-cyan-950/10">
               <div className="w-16 h-1 bg-slate-800 rounded-full overflow-hidden mb-4 relative">
                  <div className="absolute inset-y-0 left-0 bg-cyan-500 animate-[pulse_1s_ease-in-out_infinite] w-full origin-left scale-x-50"></div>
               </div>
               <p className="text-xs uppercase tracking-widest font-bold animate-pulse">Running Interventions...</p>
            </div>
          )}

          {data && !loading && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700 ease-out">
              <OutputComparison raw={data.raw_output} final={data.final_output} />
              <MetricsPanel metrics={data.metrics} interventionType={data.intervention_type} metadata={data.metadata} />
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
