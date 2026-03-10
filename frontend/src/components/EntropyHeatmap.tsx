"use client"

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export interface ProbabilisticToken {
  text: string
  entropy: number
  instability?: number | null
}

function getHighlightStyle(entropy: number): string {
  if (entropy < 0.15) {
    return "text-emerald-400 border-b-2 border-emerald-900/30"
  }
  if (entropy < 0.45) {
    return "text-amber-300 border-b-2 border-amber-900/50"
  }
  return "text-rose-500 font-bold border-b-2 border-rose-500 animate-pulse drop-shadow-[0_0_8px_rgba(244,63,94,0.3)]"
}

export default function EntropyHeatmap({
  tokens,
  className,
}: {
  tokens: ProbabilisticToken[]
  className?: string
}) {
  if (!tokens.length) {
    return (
      <div
        className={cn(
          "rounded-2xl border border-slate-800 bg-slate-950/80 p-5 text-sm text-slate-500",
          className
        )}
      >
        Token entropy is not available for this response yet.
      </div>
    )
  }

  return (
    <TooltipProvider>
      <div
        className={cn(
          "rounded-2xl border border-slate-800 bg-slate-950/80 p-8 text-lg leading-relaxed",
          className
        )}
      >
        <div className="flex flex-wrap gap-x-1 whitespace-pre-wrap rounded-xl border border-slate-900 bg-black/20 p-4 font-mono">
          {tokens.map((token, index) => (
            <Tooltip key={`${index}-${token.text}`}>
              <TooltipTrigger asChild>
                <span
                  className={cn(
                    "cursor-crosshair rounded px-0.5 transition-all hover:bg-slate-800/50",
                    getHighlightStyle(token.entropy)
                  )}
                >
                  {token.text}
                </span>
              </TooltipTrigger>
              <TooltipContent>
                <div className="space-y-1">
                  <p>
                    <span className="uppercase text-slate-500">Entropy:</span>{" "}
                    {token.entropy.toFixed(4)}
                  </p>
                  <p>
                    <span className="uppercase text-slate-500">Instability:</span>{" "}
                    {typeof token.instability === "number"
                      ? token.instability.toFixed(4)
                      : "run telemetry unavailable"}
                  </p>
                </div>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>

        <div className="mt-8 flex flex-wrap gap-6 border-t border-slate-800/50 pt-4 text-[10px] font-bold uppercase tracking-widest text-slate-500">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-400" />
            Grounded
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-amber-300" />
            Stochastic Drift
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-rose-500" />
            Hallucination Zone
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}
