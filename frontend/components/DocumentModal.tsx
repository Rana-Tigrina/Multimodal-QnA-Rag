"use client";

import { useState, useRef, useEffect } from "react";
import { IngestedDocument } from "@/types";

interface DocumentModalProps {
  isOpen: boolean;
  onClose: () => void;
  dark: boolean;
  documents: IngestedDocument[];
  onRefreshDocs: () => void;
  onDeleteDoc: (title: string) => Promise<void>;
}

export default function DocumentModal({
  isOpen,
  onClose,
  dark,
  documents,
  onRefreshDocs,
  onDeleteDoc,
}: DocumentModalProps) {
  const [activeTab, setActiveTab] = useState<"upload" | "url" | "docs">("upload");
  const [urlInput, setUrlInput] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [progressStep, setProgressStep] = useState<{ step: string; detail: string; progress: number } | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      onRefreshDocs();
      setErrorMsg("");
      setProgressStep(null);
    }
  }, [isOpen, onRefreshDocs]);

  if (!isOpen) return null;

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setIsIngesting(true);
    setErrorMsg("");
    setProgressStep({ step: "uploading", detail: `Uploading ${file.name}...`, progress: 5 });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/ingest/file`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Upload failed (HTTP ${res.status})`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "progress") {
              setProgressStep({ step: ev.step, detail: ev.detail, progress: ev.progress });
            }
          } catch {}
        }
      }

      await onRefreshDocs();
      setActiveTab("docs");
    } catch (err: any) {
      setErrorMsg(err.message || "File ingestion failed");
    } finally {
      setIsIngesting(false);
    }
  };

  const handleUrlIngest = async () => {
    const url = urlInput.trim();
    if (!url) return;

    setIsIngesting(true);
    setErrorMsg("");
    setProgressStep({ step: "scraping", detail: `Connecting to ${url}...`, progress: 10 });

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/ingest/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `URL ingestion failed (HTTP ${res.status})`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "progress") {
              setProgressStep({ step: ev.step, detail: ev.detail, progress: ev.progress });
            }
          } catch {}
        }
      }

      setUrlInput("");
      await onRefreshDocs();
      setActiveTab("docs");
    } catch (err: any) {
      setErrorMsg(err.message || "URL ingestion failed");
    } finally {
      setIsIngesting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
      <div
        className={`w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh] ${
          dark ? "bg-[#1e1e1e] border border-white/10 text-white" : "bg-white border border-gray-200 text-gray-900"
        }`}
      >
        {/* Modal Header */}
        <div className={`flex items-center justify-between px-6 py-4 border-b ${dark ? "border-white/10" : "border-gray-200"}`}>
          <div className="flex items-center gap-2">
            <span className="text-xl">📚</span>
            <h2 className="text-lg font-semibold">Knowledge Base Manager</h2>
          </div>
          <button
            onClick={onClose}
            className={`p-1.5 rounded-lg transition-colors ${
              dark ? "hover:bg-white/10 text-white/60" : "hover:bg-gray-100 text-gray-500"
            }`}
          >
            ✕
          </button>
        </div>

        {/* Tab Navigation */}
        <div className={`flex border-b px-6 gap-6 text-sm font-medium ${dark ? "border-white/10" : "border-gray-200"}`}>
          <button
            onClick={() => setActiveTab("upload")}
            className={`py-3 border-b-2 transition-all ${
              activeTab === "upload"
                ? "border-blue-500 text-blue-500 font-semibold"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            📁 Upload File
          </button>
          <button
            onClick={() => setActiveTab("url")}
            className={`py-3 border-b-2 transition-all ${
              activeTab === "url"
                ? "border-blue-500 text-blue-500 font-semibold"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            🌐 Paste Web URL
          </button>
          <button
            onClick={() => setActiveTab("docs")}
            className={`py-3 border-b-2 transition-all flex items-center gap-1.5 ${
              activeTab === "docs"
                ? "border-blue-500 text-blue-500 font-semibold"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            📄 Active Docs
            <span className="px-2 py-0.5 text-xs rounded-full bg-blue-500/20 text-blue-400 font-bold">
              {documents.length}
            </span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto flex-1">
          {errorMsg && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              ⚠️ {errorMsg}
            </div>
          )}

          {isIngesting && progressStep && (
            <div className="mb-6 p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 flex flex-col gap-2">
              <div className="flex justify-between text-xs font-semibold text-blue-400">
                <span>Processing RAG Pipeline...</span>
                <span>{progressStep.progress}%</span>
              </div>
              <div className="w-full h-2 bg-blue-950 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${progressStep.progress}%` }}
                />
              </div>
              <p className="text-xs text-blue-300/80">{progressStep.detail}</p>
            </div>
          )}

          {/* TAB 1: UPLOAD FILE */}
          {activeTab === "upload" && (
            <div className="flex flex-col gap-4">
              <div
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all ${
                  dark
                    ? "border-white/20 hover:border-blue-500 hover:bg-white/5"
                    : "border-gray-300 hover:border-blue-500 hover:bg-blue-50/50"
                }`}
              >
                <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400 text-2xl">
                  📤
                </div>
                <div className="text-center">
                  <p className="font-medium text-sm">Click or drag & drop file to upload</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Supports PDF, DOCX, TXT, Markdown, PNG, JPG, JPEG
                  </p>
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFileUpload(file);
                  }}
                  accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg"
                  className="hidden"
                />
              </div>
            </div>
          )}

          {/* TAB 2: PASTE URL */}
          {activeTab === "url" && (
            <div className="flex flex-col gap-4">
              <label className="text-xs font-medium text-gray-400">Public Web Page URL</label>
              <div className="flex gap-2">
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://example.com/documentation"
                  className={`flex-1 px-4 py-2.5 rounded-xl border text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    dark ? "bg-black/30 border-white/10 text-white" : "bg-gray-50 border-gray-300 text-gray-900"
                  }`}
                  disabled={isIngesting}
                />
                <button
                  onClick={handleUrlIngest}
                  disabled={isIngesting || !urlInput.trim()}
                  className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-all disabled:opacity-50"
                >
                  Ingest URL
                </button>
              </div>
            </div>
          )}

          {/* TAB 3: ACTIVE DOCS */}
          {activeTab === "docs" && (
            <div className="flex flex-col gap-3">
              {documents.length === 0 ? (
                <div className="text-center py-8 text-gray-400 text-sm">
                  No documents indexed yet. Upload a file or paste a web URL to build your Knowledge Base.
                </div>
              ) : (
                documents.map((doc) => (
                  <div
                    key={doc.doc_title}
                    className={`flex items-center justify-between p-3.5 rounded-xl border ${
                      dark ? "bg-white/5 border-white/10" : "bg-gray-50 border-gray-200"
                    }`}
                  >
                    <div className="flex flex-col min-w-0 pr-4">
                      <span className="font-medium text-sm truncate">{doc.doc_title}</span>
                      <span className="text-xs text-gray-400 truncate mt-0.5">{doc.source_url}</span>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      <span className="px-2.5 py-1 text-xs rounded-full bg-blue-500/10 text-blue-400 font-medium">
                        {doc.chunk_count} chunks
                      </span>
                      <button
                        onClick={() => onDeleteDoc(doc.doc_title)}
                        className="p-1.5 rounded-lg hover:bg-red-500/20 text-red-400 transition-colors text-xs"
                        title="Delete Document"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
