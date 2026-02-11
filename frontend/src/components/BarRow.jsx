import React from "react";
import { ChordCell } from "./ChordCell";

export function BarRow({ 
  barIndex, 
  BEATS_PER_BAR, 
  beats, 
  onBeatChange, 
  onRemove, 
  canRemove, 
  clipboard, 
  onCopy, 
  onPaste,
  onCopyBar,
  onPasteBar,
  hasBarClipboard
}) {
  return (
    <div className="flex items-stretch bg-gray-900/50 rounded-lg border border-gray-800 overflow-hidden">
      {/* Bar label + bar copy/paste */}
      <div className="flex flex-col items-center justify-center px-3 py-2 bg-gray-800/50 border-r border-gray-700 min-w-[70px]">
        <span className="text-[10px] text-gray-500 uppercase tracking-wide">Bar</span>
        <span className="text-lg font-semibold text-gray-300">{barIndex + 1}</span>
        
        {/* Bar copy/paste buttons */}
        <div className="flex gap-1 mt-1.5">
          <button
            onClick={() => onCopyBar(barIndex)}
            className="text-[9px] px-1.5 py-0.5 rounded bg-gray-700 border border-gray-600
                       text-gray-400 hover:bg-gray-600 hover:text-gray-300 transition"
            title="Copy entire bar"
          >
            Copy
          </button>
          <button
            onClick={() => onPasteBar(barIndex)}
            disabled={!hasBarClipboard}
            className="text-[9px] px-1.5 py-0.5 rounded bg-gray-700 border border-gray-600
                       text-gray-400 hover:bg-gray-600 hover:text-gray-300 transition
                       disabled:opacity-30 disabled:cursor-not-allowed"
            title="Paste entire bar"
          >
            Paste
          </button>
        </div>
      </div>

      {/* Beats */}
      <div className="flex-1 grid grid-cols-4">
        {beats.map((beat, i) => {
          const globalIndex = barIndex * BEATS_PER_BAR + i;
          return (
            <ChordCell
              key={`${barIndex}-${i}`}
              beat={beat}
              isFirst={i === 0}
              onChange={(updates) => onBeatChange(globalIndex, updates)}
              onCopy={() => onCopy(globalIndex)}
              onPaste={() => onPaste(globalIndex)}
              hasClipboard={clipboard !== null && !Array.isArray(clipboard)}
            />
          );
        })}
      </div>

      {/* Remove button */}
      <button
        onClick={() => onRemove(barIndex)}
        disabled={!canRemove}
        className="px-3 text-gray-500 hover:text-red-400 hover:bg-red-900/20 
                   transition disabled:opacity-20 disabled:cursor-not-allowed
                   border-l border-gray-700"
        title="Remove bar"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}