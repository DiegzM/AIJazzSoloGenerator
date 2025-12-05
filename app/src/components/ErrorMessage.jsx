import React, { useState, useEffect } from "react";

export default function ErrorMessage({ message, onDismiss }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (message) {
      // Trigger enter animation
      requestAnimationFrame(() => setIsVisible(true));

      // Auto-dismiss after 5 seconds
      const timer = setTimeout(() => {
        setIsVisible(false);
        // Wait for exit animation before clearing
        setTimeout(() => onDismiss(), 300);
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [message, onDismiss]);

  if (!message) return null;

  return (
    <div
      className={`
        fixed bottom-6 left-1/2 -translate-x-1/2 z-50
        bg-gray-900/95 border border-red-900/50 rounded-xl
        px-5 py-3 shadow-lg shadow-black/30
        transition-all duration-300 ease-out
        ${isVisible 
          ? "translate-y-0 opacity-100" 
          : "translate-y-8 opacity-0"
        }
      `}
    >
      <p className="text-red-400 text-lg">
        <span className="font-semibold">Error:</span> {message}
      </p>
    </div>
  );
}