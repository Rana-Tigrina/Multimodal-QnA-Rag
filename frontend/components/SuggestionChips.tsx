"use client";

const SUGGESTIONS = [
  "Summarize the key points in my documents",
  "What are the main requirements or rules mentioned?",
  "Extract tables and key statistics from the text",
  "What are the important dates or deadlines?",
  "Can you list the main topics covered?",
];

interface SuggestionChipsProps {
  dark: boolean;
  onSelect: (q: string) => void;
  onOpenModal: () => void;
}

export default function SuggestionChips({ dark, onSelect, onOpenModal }: SuggestionChipsProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 px-4 py-10 animate-fadeUp">
      <h1 className={`text-2xl sm:text-3xl font-semibold text-center leading-snug ${dark ? "text-white/90" : "text-gray-800"}`}>
        Universal AI <span className="text-blue-500">Knowledge Assistant</span>
      </h1>
      <p className={`text-sm text-center max-w-md leading-relaxed ${dark ? "text-white/45" : "text-gray-500"}`}>
        Upload PDFs, Word docs, text files, or web page URLs into your Knowledge Base — and get accurate, cited answers.
      </p>

      {/* Prominent Action Buttons */}
      <div className="flex gap-3 my-2">
        <button
          onClick={onOpenModal}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-all shadow-md active:scale-95"
        >
          <span>📁 Upload Document</span>
        </button>
        <button
          onClick={onOpenModal}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl border text-sm font-medium transition-all active:scale-95 ${
            dark
              ? "bg-white/5 border-white/15 text-white/90 hover:bg-white/10"
              : "bg-gray-100 border-gray-300 text-gray-800 hover:bg-gray-200"
          }`}
        >
          <span>🌐 Paste Web URL</span>
        </button>
      </div>

      <div className="flex flex-wrap gap-2 justify-center max-w-xl">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            onClick={() => onSelect(s)}
            className={`px-4 py-2 text-sm rounded-full border transition-all active:scale-95 hover:-translate-y-0.5 ${
              dark
                ? "bg-white/5 border-white/10 text-white/50 hover:bg-white/10 hover:text-white/80 hover:border-blue-500"
                : "bg-black/4 border-black/10 text-gray-500 hover:bg-black/8 hover:text-gray-700 hover:border-blue-500"
            }`}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}