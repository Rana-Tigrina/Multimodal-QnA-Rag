"use client";

import { useRef, useEffect, useState } from "react";

interface InputBarProps {
  dark: boolean;
  isLoading: boolean;
  onSend: (q: string) => void;
  onOpenModal: () => void;
}

export default function InputBar({ dark, isLoading, onSend, onOpenModal }: InputBarProps) {
  const [question, setQuestion] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX = 500;

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [question]);

  const handleSend = () => {
    if (!question.trim() || isLoading) return;
    onSend(question);
    setQuestion("");
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const remaining = MAX - question.length;
  const nearLimit = remaining < 50;

  return (
    <div className={`px-4 pb-safe pb-5 pt-3 flex-shrink-0 ${dark ? "bg-[#212121]" : "bg-white"}`}>
      <div className="max-w-3xl mx-auto">
        <div className={`relative rounded-2xl border transition-all flex items-center ${
          dark
            ? "bg-[#2f2f2f] border-white/10 focus-within:border-white/25"
            : "bg-gray-100 border-black/10 focus-within:border-black/25"
        }`}>
          {/* Quick Action Buttons (Upload File & Paste URL) */}
          <div className="flex items-center gap-1 pl-3 pr-1 flex-shrink-0">
            <button
              onClick={onOpenModal}
              title="Upload Document (PDF, DOCX, TXT, MD, Images)"
              className={`p-2 rounded-xl transition-all active:scale-90 ${
                dark ? "hover:bg-white/10 text-white/60 hover:text-white" : "hover:bg-gray-200 text-gray-500 hover:text-gray-900"
              }`}
            >
              📎
            </button>
            <button
              onClick={onOpenModal}
              title="Paste Web URL"
              className={`p-2 rounded-xl transition-all active:scale-90 ${
                dark ? "hover:bg-white/10 text-white/60 hover:text-white" : "hover:bg-gray-200 text-gray-500 hover:text-gray-900"
              }`}
            >
              🌐
            </button>
          </div>

          <textarea
            ref={textareaRef}
            rows={1}
            placeholder="Ask anything or upload documents / links..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKey}
            disabled={isLoading}
            maxLength={MAX}
            className={`w-full bg-transparent border-none outline-none resize-none px-2 py-3.5 pr-14 text-sm leading-relaxed max-h-48 overflow-y-auto ${
              dark ? "text-white/90 placeholder:text-white/30" : "text-gray-800 placeholder:text-gray-400"
            } disabled:opacity-50 text-base sm:text-sm`}
          />
          <button
            onClick={handleSend}
            disabled={!question.trim() || isLoading}
            className="absolute bottom-2.5 right-2.5 w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center transition-all active:scale-90 hover:bg-blue-500 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="white">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>

        {/* Footer */}
        <div className="flex justify-between items-center mt-1.5 px-1">
          <span className={`text-xs ${dark ? "text-white/30" : "text-gray-400"}`}>
            Click 📎 to Upload Files or 🌐 to Paste URLs · Enter to send
          </span>
          {nearLimit && (
            <span className={`text-xs ${remaining < 20 ? "text-red-400" : dark ? "text-white/30" : "text-gray-400"}`}>
              {remaining}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}