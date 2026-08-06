"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message } from "@/types";

interface MessageBubbleProps {
  message: Message;
  dark: boolean;
  onRegenerate: (id: number) => void;
  isLoading: boolean;
}

function MarkdownContent({ content, dark }: { content: string; dark: boolean }) {
  const clean = content.replace(/\n*📎[\s\S]*$/m, "").trim();
  return (
    <div className={`prose prose-sm max-w-none [&_a]:text-blue-500 [&_a]:no-underline hover:[&_a]:underline ${dark ? "prose-invert" : ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{clean}</ReactMarkdown>
    </div>
  );
}

export default function MessageBubble({ message, dark, onRegenerate, isLoading }: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isBot = message.role === "assistant";

  return (
    <div className="max-w-3xl mx-auto px-4 py-3 animate-fadeUp group">
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0 ${
          isBot ? "bg-blue-600 text-white" : dark ? "bg-white/20 text-white/80" : "bg-gray-200 text-gray-600"
        }`}>
          {isBot ? "AI" : "U"}
        </div>
        <span className={`text-xs font-semibold ${dark ? "text-white/45" : "text-gray-400"}`}>
          {isBot ? "AI Assistant" : "You"}
        </span>
      </div>

      {/* Content */}
      <div className={`pl-9 text-sm leading-relaxed ${
        message.error ? "text-red-400" :
        isBot ? (dark ? "text-white/90" : "text-gray-800") :
        (dark ? "text-white/70" : "text-gray-700")
      }`}>
        {message.loading && !message.content ? (
          message.status ? (
            <div className={`flex items-center gap-2 text-xs italic ${dark ? "text-white/30" : "text-gray-400"}`}>
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              {message.status}
              <div className="flex gap-1">
                {[0, 1, 2].map(i => (
                  <span key={i} className={`w-1.5 h-1.5 rounded-full ${dark ? "bg-white/20" : "bg-gray-300"} animate-bounce`}
                    style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          ) : (
            <div className="flex gap-1">
              {[0, 1, 2].map(i => (
                <span key={i} className={`w-1.5 h-1.5 rounded-full ${dark ? "bg-white/20" : "bg-gray-300"} animate-bounce`}
                  style={{ animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
          )
        ) : isBot ? (
          <MarkdownContent content={message.content} dark={dark} />
        ) : (
          message.content
        )}
      </div>

      {/* Sources */}
      {isBot && !message.loading && message.sources && message.sources.length > 0 && (
        <div className="pl-9 mt-2 flex flex-col gap-1">
          {message.sources.filter(s => s.url).map((s, i) => (
            <a key={i} href={s.url} target="_blank" rel="noopener noreferrer"
              className="text-xs text-blue-500 hover:underline opacity-80 hover:opacity-100 transition-opacity break-all">
              📎 {s.section ? `${s.section} — ` : ""}{s.doc || s.text || "Source"}
            </a>
          ))}
        </div>
      )}


      {/* Actions — show on every bot message */}
      {isBot && !message.loading && message.content && (
        <div className="pl-9 mt-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {/* Copy */}
          <button
            onClick={copy}
            title="Copy"
            className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border transition-all active:scale-95 ${
              copied
                ? "border-green-500 text-green-500"
                : dark
                  ? "border-white/15 text-white/40 hover:bg-white/10 hover:text-white/70"
                  : "border-black/12 text-gray-400 hover:bg-black/6 hover:text-gray-600"
            }`}
          >
            {copied ? (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
                Copied
              </>
            ) : (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Copy
              </>
            )}
          </button>

          {/* Regenerate */}
          <button
            onClick={() => onRegenerate(message.id)}
            disabled={isLoading}
            title="Regenerate"
            className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border transition-all active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed ${
              dark
                ? "border-white/15 text-white/40 hover:bg-white/10 hover:text-white/70"
                : "border-black/12 text-gray-400 hover:bg-black/6 hover:text-gray-600"
            }`}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              className={isLoading ? "animate-spin" : ""}>
              <polyline points="23 4 23 10 17 10"/>
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
            </svg>
            Regenerate
          </button>
        </div>
      )}
    </div>
  );
}