import React from "react";

export default function OutputComparison({ raw, final }: any) {
  const highlightDiff = (rawStr: string, finalStr: string) => {
    if (!rawStr || !finalStr) return finalStr;
    const rawTokens = rawStr.split(" ");
    const finalTokens = finalStr.split(" ");
    
    return finalTokens.map((t, idx) => {
      // Very basic diff detection for visual flair
      // Standardizes casing and punctuation simple drops for matching
      const cleanT = t.replace(/[.,!?]/g, "");
      const isNew = !rawStr.includes(cleanT);
      
      if (isNew) {
        return (
          <span key={idx} className="bg-green-900 text-green-300 rounded px-1 transition-all duration-300">
            {t}
          </span>
        );
      }
      return <span key={idx} className="text-gray-300">{t}</span>;
    });
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="p-5 border border-gray-700/50 rounded-lg bg-gray-900/50">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-4 flex items-center">
          <span className="w-2 h-2 rounded-full bg-gray-600 mr-2"></span>
          Pre-Rescue (Model Output)
        </h3>
        <p className="text-gray-400 leading-relaxed font-serif tracking-wide">{raw}</p>
      </div>

      <div className="p-5 border border-blue-800/50 rounded-lg bg-[#0a1128] shadow-[0_0_20px_rgba(30,58,138,0.15)]">
        <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-widest mb-4 flex items-center">
          <span className="w-2 h-2 rounded-full bg-blue-500 mr-2 animate-pulse"></span>
          Post-Rescue (Pipeline Output)
        </h3>
        <p className="leading-relaxed font-serif tracking-wide flex flex-wrap gap-x-1 gap-y-1">
          {highlightDiff(raw, final)}
        </p>
      </div>
    </div>
  );
}
