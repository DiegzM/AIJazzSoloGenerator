import React from "react";
import { ROOT_MAP, GUI_ROOTS, GUI_QUALITIES } from "../constants/definitions";

export function ChordCell({ beat, onChange, isFirst, onCopy, onPaste, hasClipboard }) {
  const isNoChord = !beat.root || beat.guiRoot === "NC";
  const displayText = isNoChord ? "—" : `${beat.guiRoot}${beat.guiQuality}`;

  const handleRootChange = (value) => {
    if (value === "NC") {
      onChange({ guiRoot: "NC", guiQuality: "NC", root: "", quality_class: "" });
    } else {
      const newRoot = ROOT_MAP[value];
      // If switching from NC, default to maj
      const newQuality = beat.guiQuality === "NC" ? "maj" : beat.guiQuality;
      const qualityObj = GUI_QUALITIES.find(q => q.label === newQuality) || GUI_QUALITIES[0];
      onChange({
        guiRoot: value,
        guiQuality: newQuality,
        root: newRoot,
        quality_class: qualityObj.value
      });
    }
  };

  const handleQualityChange = (value) => {
    if (value === "NC") {
      onChange({ guiRoot: "NC", guiQuality: "NC", root: "", quality_class: "" });
    } else {
      const qualityObj = GUI_QUALITIES.find(q => q.label === value);
      onChange({
        ...beat,
        guiQuality: value,
        quality_class: qualityObj.value
      });
    }
  };

  return (
    <div className={`flex flex-col gap-1.5 p-2 ${!isFirst ? "border-l border-gray-700" : ""}`}>
      <span className="text-[10px] text-gray-500 uppercase tracking-wide">Beat {beat.beat}</span>
      
      <div className={`text-center py-1 px-2 rounded text-sm font-medium ${
        isNoChord ? "text-gray-500" : "text-amber-300"
      }`}>
        {displayText}
      </div>

      <select
        value={beat.guiRoot || "NC"}
        onChange={(e) => handleRootChange(e.target.value)}
        className="w-full text-xs bg-gray-800/50 border border-gray-700 rounded px-1.5 py-1 
                   text-gray-300 focus:border-amber-500 focus:outline-none transition"
      >
        <option value="NC">No Chord</option>
        {GUI_ROOTS.map(r => <option key={r} value={r}>{r}</option>)}
      </select>

      <select
        value={beat.guiQuality || "NC"}
        onChange={(e) => handleQualityChange(e.target.value)}
        disabled={isNoChord}
        className="w-full text-xs bg-gray-800/50 border border-gray-700 rounded px-1.5 py-1 
                   text-gray-300 focus:border-amber-500 focus:outline-none transition
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        <option value="NC">—</option>
        {GUI_QUALITIES.map(q => (
          <option key={q.label} value={q.label}>{q.label}</option>
        ))}
      </select>

      {/* Copy/Paste buttons */}
      <div className="flex gap-1 mt-1">
        <button
          onClick={(e) => { e.stopPropagation(); onCopy(); }}
          disabled={isNoChord}
          className="flex-1 text-[10px] px-1.5 py-1 rounded bg-gray-800 border border-gray-700
                     text-gray-400 hover:bg-gray-700 hover:text-gray-300 transition
                     disabled:opacity-30 disabled:cursor-not-allowed"
          title="Copy chord"
        >
          Copy
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onPaste(); }}
          disabled={!hasClipboard}
          className="flex-1 text-[10px] px-1.5 py-1 rounded bg-gray-800 border border-gray-700
                     text-gray-400 hover:bg-gray-700 hover:text-gray-300 transition
                     disabled:opacity-30 disabled:cursor-not-allowed"
          title="Paste chord"
        >
          Paste
        </button>
      </div>
    </div>
  );
}