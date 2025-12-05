import React from "react";
import { KEY_OPTIONS } from "../constants/definitions";

export default function SoloControls({ controls, setControls }) {
  const update = (field, value) => {
    setControls({ ...controls, [field]: value });
  };

  return (
    <div className="w-full mb-8 p-6 bg-transparent shadow-md rounded-xl border border-gray-700 text-white flex flex-col items-center">
      <h2 className="text-3xl p-2 font-semibold mb-4">SOLO SETTINGS</h2>

      {/* FILE NAME (max 100 characters)*/}
      <div className="mb-4 w-80">
        <label className="block text-sm font-medium text-gray-400 mb-2">
          Output File Name (max 100 characters)
        </label>
        <input
          type="text"
          value={controls.outputFileName}
          onChange={(e) => update("outputFileName", e.target.value)}
          placeholder={controls.outputFileName}
          className="w-full rounded-md 
                    bg-[#131a22] text-gray-200 
                    border border-gray-600 
                    px-3 py-2
                    focus:ring-amber-400 focus:border-amber-400 
                    appearance-none"
          maxLength={100}
        />
      </div>
      {/* TEMPO */}
      <div className="mb-4 w-64">
        <label className="block text-sm font-medium text-gray-400 mb-2">
          Tempo: <span className="text-amber-400 font-semibold">{controls.avgtempo} BPM</span>
        </label>
        <input
          type="range"
          min={60}
          max={300}
          step={5}
          value={controls.avgtempo}
          onChange={(e) => update("avgtempo", parseInt(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                     accent-amber-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>60</span>
          <span>300</span>
        </div>
      </div>

      {/* STYLE */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-400">
          Style
        </label>
        <select
          value={controls.style}
          onChange={(e) => update("style", e.target.value)}
          className="mt-1 w-40 rounded-md 
                    bg-[#131a22] text-gray-200 
                    border border-gray-600 
                    px-3 py-2
                    focus:ring-amber-400 focus:border-amber-400 
                    appearance-none
                    cursor-pointer"
        >

          <option value="BEBOP">BEBOP</option>
          <option value="POSTBOP">POSTBOP</option>
          <option value="HARDBOP">HARDBOP</option>
          <option value="SWING">SWING</option>
          <option value="COOL">COOL</option>
        </select>
      </div>

      {/* KEY */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-400">
          Key
        </label>
        <select
          value={controls.key}
          onChange={(e) => update("key", e.target.value)}
          className="mt-1 w-40 rounded-md 
                    bg-[#131a22] text-gray-200 
                    border border-gray-600 
                    px-3 py-2
                    focus:ring-amber-400 focus:border-white-400 
                    appearance-none
                    cursor-pointer"
        >

          {Object.entries(KEY_OPTIONS).map(([label, value]) => (
            <option key={label} value={value}>
              {label}
            </option>
          ))}

        </select>
      </div>
    </div>
  );
}
