"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Message, Source } from "@/types";

interface MessageBubbleProps {
  message: Message;
  dark: boolean;
  onRegenerate: (id: number) => void;
  isLoading: boolean;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function MarkdownContent({ content, dark }: { content: string; dark: boolean }) {
  return (
    <div className={`prose prose-sm max-w-none [&_a]:text-blue-500 [&_a]:no-underline hover:[&_a]:underline ${dark ? "prose-invert" : ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function CitationCard({ source, dark }: { source: Source; dark: boolean }) {
  const [expanded, setExpanded] = useState(false);

  const isWebUrl = source.url && (source.url.startsWith("http://") || source.url.startsWith("https://"));
  const isFile = source.url && source.url.startsWith("file://");
  const fileName = isFile ? source.url.replace("file://", "").trim() : "";
  const fileDownloadUrl = isFile ? `${API_URL}/files/${encodeURIComponent(fileName)}` : source.url;

  const typeIcon = source.type === "image" ? "🖼️" : isWebUrl ? "🌐" : "📄";

  return (
    <div
      className={`rounded-xl border transition-all text-xs overflow-hidden ${
        dark
          ? "bg-[#282828] border-white/10 hover:border-white/20"
          : "bg-gray-50 border-gray-200 hover:border-gray-300"
      }`}
    >
      <div className="flex items-center justify-between px-3 py-2 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm flex-shrink-0">{typeIcon}</span>
          <div className="truncate">
            <span className={`font-semibold ${dark ? "text-white/90" : "text-gray-900"}`}>
              {source.doc || "Document"}
            </span>
            {source.section && (
              <span className={`ml-1.5 opacity-60 ${dark ? "text-white" : "text-gray-700"}`}>
                • {source.section}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1.5 flex-shrink-0">
          {source.snippet && (
            <button
              onClick={() => setExpanded(!expanded)}
              className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-all ${
                expanded
                  ? dark
                    ? "bg-blue-500/20 border-blue-500/40 text-blue-400"
                    : "bg-blue-50 border-blue-200 text-blue-600"
                  : dark
                  ? "bg-white/5 border-white/10 text-white/70 hover:bg-white/10"
                  : "bg-white border-gray-200 text-gray-600 hover:bg-gray-100"
              }`}
            >
              {expanded ? "Hide Quote" : "View Excerpt"}
            </button>
          )}

          {isWebUrl && (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className={`px-2 py-0.5 rounded text-[11px] font-medium border flex items-center gap-1 transition-all ${
                dark
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20"
                  : "bg-blue-50 border-blue-200 text-blue-600 hover:bg-blue-100"
              }`}
              title={source.url}
            >
              <span>Visit Link</span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                <polyline points="15 3 21 3 21 9" />
                <line x1="10" y1="14" x2="21" y2="3" />
              </svg>
            </a>
          )}

          {isFile && fileName && (
            <a
              href={fileDownloadUrl}
              target="_blank"
              rel="noopener noreferrer"
              download={fileName}
              className={`px-2 py-0.5 rounded text-[11px] font-medium border flex items-center gap-1 transition-all ${
                dark
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
                  : "bg-emerald-50 border-emerald-200 text-emerald-600 hover:bg-emerald-100"
              }`}
              title={`Download or view ${fileName}`}
            >
              <span>Download File</span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </a>
          )}
        </div>
      </div>

      {expanded && source.snippet && (
        <div
          className={`px-3 py-2 border-t font-mono text-[11px] leading-relaxed transition-all ${
            dark
              ? "bg-[#1f1f1f] border-white/5 text-white/80"
              : "bg-white border-gray-100 text-gray-700"
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider mb-1 opacity-50">
            Source Excerpt / Match:
          </div>
          &ldquo;{source.snippet}&rdquo;
        </div>
      )}
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

      {/* Rich Sources & Citations */}
      {isBot && !message.loading && message.sources && message.sources.length > 0 && (
        <div className="pl-9 mt-3 flex flex-col gap-2">
          <div className={`text-[11px] font-bold uppercase tracking-wider ${dark ? "text-white/40" : "text-gray-400"}`}>
            Verified Citations ({message.sources.length})
          </div>
          <div className="flex flex-col gap-1.5">
            {message.sources.map((source, i) => (
              <CitationCard key={i} source={source} dark={dark} />
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      {isBot && !message.loading && message.content && (
        <div className="pl-9 mt-2.5 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {/* Copy */}
          <button
            onClick={copy}
            title="Copy response"
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