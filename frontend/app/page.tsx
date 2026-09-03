"use client";

import TopBar from "@/components/TopBar";
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import DocumentModal from "@/components/DocumentModal";
import { useChat } from "@/hooks/useChat";

export default function Home() {
  const {
    messages,
    isLoading,
    dark,
    setDark,
    documents,
    isModalOpen,
    setIsModalOpen,
    fetchDocuments,
    deleteDocument,
    sendQuestion,
    regenerate,
    clearChat,
    selectedModel,
    setSelectedModel,
  } = useChat();

  return (
    <div
      style={{ height: "100dvh" }}
      className={`flex flex-col w-screen overflow-hidden ${dark ? "bg-[#212121]" : "bg-white"}`}
    >
      <TopBar
        dark={dark}
        setDark={setDark}
        hasMessages={messages.length > 0}
        clearChat={clearChat}
        docCount={documents.length}
        onOpenModal={() => setIsModalOpen(true)}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
      />
      <ChatWindow
        messages={messages}
        dark={dark}
        isLoading={isLoading}
        onRegenerate={regenerate}
        onSuggestion={sendQuestion}
        onOpenModal={() => setIsModalOpen(true)}
      />
      <InputBar
        dark={dark}
        isLoading={isLoading}
        onSend={sendQuestion}
        onOpenModal={() => setIsModalOpen(true)}
      />

      <DocumentModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        dark={dark}
        documents={documents}
        onRefreshDocs={fetchDocuments}
        onDeleteDoc={deleteDocument}
      />
    </div>
  );
}