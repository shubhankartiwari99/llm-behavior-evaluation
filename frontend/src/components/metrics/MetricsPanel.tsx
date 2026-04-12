import React from "react";

export default function MetricsPanel({ metrics, interventionType }: any) {
  if (!metrics) return null;

  const collapseLabel = metrics.collapse_ratio < 0.3 ? "Heavy shaping" : 
                        metrics.collapse_ratio < 0.6 ? "Moderate shaping" : "Light shaping";
  
  const collapseColor = metrics.collapse_ratio < 0.3 ? "text-red-400" :
                        metrics.collapse_ratio < 0.6 ? "text-yellow-400" : "text-green-400";

  return (
    <div className="space-y-4 p-5 border rounded-lg bg-[#080808] border-gray-800">
      <h3 className="text-md font-medium text-gray-300">System Trace Metrics</h3>
      
      {interventionType && (
        <div className="p-3 bg-blue-900/20 border border-blue-900/50 rounded-md flex items-center">
          <span className="text-blue-400 text-sm font-semibold tracking-wider uppercase mr-3">Runtime Action</span> 
          <span className="font-mono text-blue-200 text-sm px-2 py-1 bg-blue-950 rounded">{interventionType}</span>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 bg-[#111] rounded-md border border-gray-800">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Collapse Ratio</div>
          <div className="flex items-baseline space-x-2">
            <span className={`text-2xl font-light ${collapseColor}`}>{metrics.collapse_ratio.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-xs text-gray-500">{collapseLabel}</div>
        </div>

        <div className="p-4 bg-[#111] rounded-md border border-gray-800">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Stage Change</div>
          <div className="flex items-baseline space-x-2">
             <span className="text-2xl font-light text-gray-300">{(metrics.stage_change_rate * 100).toFixed(0)}%</span>
          </div>
          <div className="mt-1 text-xs text-gray-500">Pipeline trigger rate</div>
        </div>

        <div className="p-4 bg-[#111] rounded-md border border-gray-800">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Raw Entropy</div>
          <div className="flex items-baseline space-x-2">
             <span className="text-2xl font-light text-gray-300">{metrics.entropy_raw.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-xs text-gray-500">Pre-intervention</div>
        </div>

        <div className="p-4 bg-[#111] rounded-md border border-gray-800">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Final Entropy</div>
          <div className="flex items-baseline space-x-2">
             <span className="text-2xl font-light text-gray-300">{metrics.entropy_final.toFixed(2)}</span>
          </div>
          <div className="mt-1 text-xs text-gray-500">Post-intervention</div>
        </div>
      </div>

      <div className="mt-4 p-4 bg-gray-900 border border-gray-700/50 rounded-lg">
        <h4 className="text-xs text-gray-400 uppercase tracking-widest mb-1">System Insight</h4>
        <p className="text-sm text-gray-300 leading-relaxed">
          Runtime intervention reduced output entropy by ~{((1 - metrics.collapse_ratio) * 100).toFixed(0)}%, 
          constraining the response distribution into a structured manifold while preserving semantic coherence.
        </p>
      </div>
    </div>
  );
}
