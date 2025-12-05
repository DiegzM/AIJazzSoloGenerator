export default function GenerateButton({ onGenerate, generating }) {
  return (
    <button
      onClick={onGenerate}
      disabled={generating}
      className={`
        px-8 py-3 rounded-xl font-semibold text-lg transition-all duration-200
        ${generating 
          ? "bg-gray-700 text-gray-400 cursor-not-allowed" 
          : "bg-amber-500 hover:bg-amber-400 text-gray-900"
        }
      `}
    >
      {generating ? (
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Generating...
        </span>
      ) : (
        "Generate Solo"
      )}
    </button>
  );
}