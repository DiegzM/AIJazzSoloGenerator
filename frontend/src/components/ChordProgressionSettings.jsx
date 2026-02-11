import React, { useState, useEffect } from "react";
import { BarRow } from "./BarRow";

const BEATS_PER_BAR = 4;
const MAX_BARS = 100;

export default function ChordProgressionControls({ beats, setBeats, onError }) {
  const [bars, setBars] = useState(2);
  const [chordClipboard, setChordClipboard] = useState(null); // { guiRoot, guiQuality, root, quality_class }
  const [barClipboard, setBarClipboard] = useState(null); // array of 4 chord objects

  const makeEmptyBeat = (bar, beat) => ({
    bar,
    beat,
    root: "",
    quality_class: "",
    guiRoot: "NC",
    guiQuality: "NC"
  });

  // Initialize beats
  useEffect(() => {
    if (beats.length === 0) {
      const arr = [];
      for (let bar = 0; bar < bars; bar++) {
        for (let beat = 1; beat <= BEATS_PER_BAR; beat++) {
          arr.push(makeEmptyBeat(bar, beat));
        }
      }
      setBeats(arr);
    }
  }, []);

  // Sync bars count with beats array
  useEffect(() => {
    if (beats.length > 0) {
      setBars(Math.ceil(beats.length / BEATS_PER_BAR));
    }
  }, [beats.length]);

  const addBar = () => {
    if (bars >= MAX_BARS) {
      onError(`Maximum of ${MAX_BARS} bars reached.`);
      return;
    };
    const newBarIndex = bars;
    const newBeats = [];
    for (let beat = 1; beat <= BEATS_PER_BAR; beat++) {
      newBeats.push(makeEmptyBeat(newBarIndex, beat));
    }
    setBeats([...beats, ...newBeats]);
    setBars(bars + 1);
  };

  const removeBar = (barIndex) => {
    if (bars <= 1) return;
    const startIdx = barIndex * BEATS_PER_BAR;
    const newBeats = [
      ...beats.slice(0, startIdx),
      ...beats.slice(startIdx + BEATS_PER_BAR)
    ].map((b, i) => ({
      ...b,
      bar: Math.floor(i / BEATS_PER_BAR),
      beat: (i % BEATS_PER_BAR) + 1
    }));
    setBeats(newBeats);
    setBars(bars - 1);
  };

  // when loop button is pressed, duplicate the current beats once and append to the end
  const loop = () => {
    const newBeats = [...beats, ...beats.map(b => ({ ...b, bar: b.bar + bars }))];
    setBeats(newBeats);
    setBars(bars * 2);
  };

  const updateBeat = (index, updates) => {
    const newBeats = [...beats];
    newBeats[index] = { ...newBeats[index], ...updates };
    setBeats(newBeats);
  };

  // Chord copy/paste (single cell)
  const copyChord = (index) => {
    const beat = beats[index];
    if (beat.root && beat.guiRoot !== "NC") {
      setBarClipboard(null); // Clear bar clipboard
      setChordClipboard({
        guiRoot: beat.guiRoot,
        guiQuality: beat.guiQuality,
        root: beat.root,
        quality_class: beat.quality_class
      });
    }
  };

  const pasteChord = (index) => {
    if (!chordClipboard) return;
    updateBeat(index, {
      guiRoot: chordClipboard.guiRoot,
      guiQuality: chordClipboard.guiQuality,
      root: chordClipboard.root,
      quality_class: chordClipboard.quality_class
    });
  };

  // Bar copy/paste (entire bar)
  const copyBar = (barIndex) => {
    const startIdx = barIndex * BEATS_PER_BAR;
    const barBeats = beats.slice(startIdx, startIdx + BEATS_PER_BAR);
    setChordClipboard(null); // Clear chord clipboard
    setBarClipboard(barBeats.map(beat => ({
      guiRoot: beat.guiRoot,
      guiQuality: beat.guiQuality,
      root: beat.root,
      quality_class: beat.quality_class
    })));
  };

  const pasteBar = (barIndex) => {
    if (!barClipboard || barClipboard.length !== BEATS_PER_BAR) return;
    const startIdx = barIndex * BEATS_PER_BAR;
    const newBeats = [...beats];
    for (let i = 0; i < BEATS_PER_BAR; i++) {
      newBeats[startIdx + i] = {
        ...newBeats[startIdx + i],
        guiRoot: barClipboard[i].guiRoot,
        guiQuality: barClipboard[i].guiQuality,
        root: barClipboard[i].root,
        quality_class: barClipboard[i].quality_class
      };
    }
    setBeats(newBeats);
  };

  // Loading state
  if (beats.length !== bars * BEATS_PER_BAR) {
    return (
      <div className="w-full p-6 border border-gray-800 rounded-xl bg-gray-900/30">
        <div className="text-gray-500 text-center">Loading chord grid…</div>
      </div>
    );
  }

  // Group beats by bar
  const barGroups = [];
  for (let i = 0; i < bars; i++) {
    barGroups.push(beats.slice(i * BEATS_PER_BAR, (i + 1) * BEATS_PER_BAR));
  }

  // Build clipboard display text
  const getClipboardDisplay = () => {
    if (barClipboard) {
      const chordNames = barClipboard
        .filter(c => c.guiRoot !== "NC")
        .map(c => `${c.guiRoot}${c.guiQuality}`);
      if (chordNames.length === 0) return "Bar (empty)";
      return `Bar: ${chordNames.join(" | ")}`;
    }
    if (chordClipboard) {
      return `${chordClipboard.guiRoot}${chordClipboard.guiQuality}`;
    }
    return null;
  };

  const clipboardDisplay = getClipboardDisplay();

  return (
    <div className="w-full p-6 border border-gray-800 rounded-xl bg-transparent mb-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-semibold text-gray-200 tracking-wide">
            Chord Progression
          </h2>
          {clipboardDisplay && (
            <span className="text-xs px-2 py-0.5 rounded bg-amber-900/30 text-amber-400 border border-amber-700/30">
              Copied: {clipboardDisplay}
            </span>
          )}
        </div>
        <button
          onClick={loop}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium
                     bg-amber-600/20 text-amber-400 border border-amber-600/30 
                     rounded-lg hover:bg-amber-600/30 transition"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v16a1 1 0 01-1 1H4a1 1 0 01-1-1V4z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5-6l3 3-3 3" />
          </svg>
          Loop Progression
        </button>
        <button
          onClick={addBar}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium
                     bg-amber-600/20 text-amber-400 border border-amber-600/30 
                     rounded-lg hover:bg-amber-600/30 transition"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Add Bar
        </button>
      </div>

      {/* Bar rows in a scrollable container */}
      <div className="overflow-x-auto px-3 border border-gray-700 rounded-lg" style={{ maxHeight: "400px", scrollbarColor: "white transparent" }}>
        <div className="flex flex-col gap-3 py-3">
          {barGroups.map((barBeats, barIndex) => (
            <BarRow
              key={barIndex}
              BEATS_PER_BAR={BEATS_PER_BAR}
              barIndex={barIndex}
              beats={barBeats}
              onBeatChange={updateBeat}
              onRemove={removeBar}
              canRemove={bars > 1}
              clipboard={chordClipboard}
              onCopy={copyChord}
              onPaste={pasteChord}
              onCopyBar={copyBar}
              onPasteBar={pasteBar}
              hasBarClipboard={barClipboard !== null}
            />
          ))}
        </div>
      </div>
        
      {/* Footer hint */}
      <p className="text-xs text-gray-600 mt-4 text-center">
        {bars} bar{bars !== 1 ? "s" : ""} · {bars * BEATS_PER_BAR} beats total
      </p>
    </div>
  );
}