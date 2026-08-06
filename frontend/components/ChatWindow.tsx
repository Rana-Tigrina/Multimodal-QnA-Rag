"use client";

import { useEffect, useRef, useState } from "react";
import { Message } from "@/types";
import MessageBubble from "./MessageBubble";
import SuggestionChips from "./SuggestionChips";

interface ChatWindowProps {
  messages: Message[];
  dark: boolean;
  isLoading: boolean;
  onRegenerate: (id: number) => void;
  onSuggestion: (q: string) => void;
  onOpenModal: () => void;
}

export default function ChatWindow({ messages, dark, isLoading, onRegenerate, onSuggestion, onOpenModal }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setShowScrollBtn(distFromBottom > 200);
  };

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="relative flex-1 min-h-0">
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className={`h-full overflow-y-auto py-4 scroll-smooth ${dark ? "bg-[#212121]" : "bg-white"}`}
      >
        {messages.length === 0 ? (
          <SuggestionChips dark={dark} onSelect={onSuggestion} onOpenModal={onOpenModal} />
        ) : (

          messages.map(msg => (
            <MessageBubble
              key={msg.id}
              message={msg}
              dark={dark}
              onRegenerate={onRegenerate}
              isLoading={isLoading}
            />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Scroll to bottom button */}
      {showScrollBtn && (
        <button
          onClick={scrollToBottom}
          className={`absolute bottom-4 right-4 w-9 h-9 rounded-full shadow-lg flex items-center justify-center transition-all active:scale-95 hover:scale-105 ${
            dark ? "bg-white/15 hover:bg-white/25 text-white" : "bg-white hover:bg-gray-50 text-gray-600 border border-black/10"
          }`}
          title="Scroll to bottom"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>
      )}
    </div>
  );
}