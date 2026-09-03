"use client";

interface TopBarProps {
  dark: boolean;
  setDark: (d: boolean) => void;
  hasMessages: boolean;
  clearChat: () => void;
  docCount: number;
  onOpenModal: () => void;
  selectedModel: string;
  setSelectedModel: (m: string) => void;
}

export default function TopBar({
  dark,
  setDark,
  hasMessages,
  clearChat,
  docCount,
  onOpenModal,
  selectedModel,
  setSelectedModel,
}: TopBarProps) {
  return (
    <div className={`flex items-center justify-between px-4 py-3 border-b flex-shrink-0 ${dark ? "border-white/10 bg-[#212121]" : "border-black/8 bg-white"}`}>
      {/* Left */}
      <div className="flex items-center gap-2.5">
        <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
        <span className={`text-sm font-semibold tracking-tight ${dark ? "text-white/90" : "text-gray-900"}`}>
          Universal AI Knowledge Assistant
        </span>
      </div>

      {/* Center: Model Selector Pill */}
      <div className={`flex items-center p-0.5 rounded-xl border text-xs font-medium ${
        dark ? "bg-white/[0.04] border-white/10" : "bg-gray-100 border-gray-200"
      }`}>
        <button
          onClick={() => setSelectedModel("groq")}
          className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all ${
            selectedModel === "groq"
              ? dark
                ? "bg-purple-600 text-white shadow-sm font-semibold"
                : "bg-white text-purple-700 shadow-sm font-semibold"
              : dark
                ? "text-white/60 hover:text-white"
                : "text-gray-600 hover:text-gray-900"
          }`}
          title="Cloud inference via Groq (zero reasoning overhead, 4096 tokens)"
        >
          <span>⚡ Qwen 3.8 27B</span>
          <span className={`text-[10px] px-1 py-0.2 rounded font-bold ${
            selectedModel === "groq"
              ? dark ? "bg-white/20 text-white" : "bg-purple-100 text-purple-800"
              : "opacity-60"
          }`}>Cloud</span>
        </button>

        <button
          onClick={() => setSelectedModel("llamacpp")}
          className={`flex items-center gap-1.5 px-3 py-1 rounded-lg transition-all ${
            selectedModel === "llamacpp"
              ? dark
                ? "bg-emerald-600 text-white shadow-sm font-semibold"
                : "bg-white text-emerald-700 shadow-sm font-semibold"
              : dark
                ? "text-white/60 hover:text-white"
                : "text-gray-600 hover:text-gray-900"
          }`}
          title="100% offline llama.cpp inference using Ling-3.0-tiny GGUF (Q4_K_M)"
        >
          <span>💻 Ling 3.0 Tiny</span>
          <span className={`text-[10px] px-1 py-0.2 rounded font-bold ${
            selectedModel === "llamacpp"
              ? dark ? "bg-white/20 text-white" : "bg-emerald-100 text-emerald-800"
              : "opacity-60"
          }`}>Local</span>
        </button>
      </div>

      {/* Right */}
      <div className="flex items-center gap-2">
        {/* Knowledge Base Manager button */}
        <button
          onClick={onOpenModal}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-all active:scale-95 ${
            dark
              ? "bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20"
              : "bg-blue-50 border-blue-200 text-blue-600 hover:bg-blue-100"
          }`}
        >
          <span>📄 Knowledge Base</span>
          <span className="px-1.5 py-0.2 rounded-md bg-blue-500 text-white text-[10px] font-bold">
            {docCount}
          </span>
        </button>

        {/* Theme toggle */}
        <button
          onClick={() => setDark(!dark)}
          className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all active:scale-95 ${dark ? "hover:bg-white/10 text-white/60" : "hover:bg-black/8 text-gray-500"}`}
          title={dark ? "Light mode" : "Dark mode"}
        >
          {dark ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </button>

        {/* Clear chat */}
        {hasMessages && (
          <button
            onClick={clearChat}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-all active:scale-95 ${dark ? "border-white/15 text-white/50 hover:bg-white/10 hover:text-white/80" : "border-black/12 text-gray-500 hover:bg-black/6 hover:text-gray-700"}`}
          >
            Clear chat
          </button>
        )}
      </div>
    </div>
  );
}