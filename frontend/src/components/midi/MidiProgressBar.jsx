import { useRef } from "react";

export default function MidiProgressBar({ progress, onSeek }) {
  const barRef = useRef(null);

  const handleInteraction = (e) => {
    const rect = barRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percentage = (x / rect.width) * 100;

    onSeek(percentage);
  };

  const handleMouseDown = (e) => {
    handleInteraction(e);

    const handleMouseMove = (e) => handleInteraction(e);
    const handleMouseUp = () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  };

  return (
    <div
      ref={barRef}
      className="w-full max-w-md h-3 bg-gray-700 rounded-full cursor-pointer relative mb-4"
      onMouseDown={handleMouseDown}
    >
      {/* Fill */}
      <div
        className="h-full bg-amber-500 rounded-full"
        style={{ width: `${progress}%` }}
      />
      {/* Handle */}
      <div
        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-md"
        style={{ left: `calc(${progress}% - 8px)` }}
      />
    </div>
  );
}