"use client";
import { useState, useRef, useEffect } from "react";
import classNames from "classnames";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Message = {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
  imagePath?: string;
};

type ChatSession = {
  sessionId: string;
  messages: Message[];
  currentScan?: {
    imagePath: string;
    segmentationResults?: any;
    similarCases?: any[];
  };
};

export default function ChatInterface() {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Initialize session
    const sessionId = `session_${Date.now()}`;
    setSession({
      sessionId,
      messages: [
        {
          role: "assistant",
          content: "Hello! I'm your medical imaging diagnostic assistant. I can help you analyze MRI/CT scans. Please upload a DICOM or NIfTI file to get started, or ask me a question.",
          timestamp: new Date().toISOString(),
        },
      ],
    });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [session?.messages]);

  const handleFileUpload = async (uploadedFile: File) => {
    if (!uploadedFile) return;

    setUploading(true);
    setFile(uploadedFile);

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);

      // Add user message
      const userMessage: Message = {
        role: "user",
        content: `Uploaded file: ${uploadedFile.name}`,
        timestamp: new Date().toISOString(),
        imagePath: uploadedFile.name,
      };

      setSession((prev) => ({
        ...prev!,
        messages: [...prev!.messages, userMessage],
      }));

      // Add processing message
      const processingMessage: Message = {
        role: "assistant",
        content: "Processing your scan... This may take a moment.",
        timestamp: new Date().toISOString(),
      };

      setSession((prev) => ({
        ...prev!,
        messages: [...prev!.messages, processingMessage],
      }));

      // Process the file
      const response = await fetch(`${API_URL}/process-upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process file");
      }

      const result = await response.json();

      // Update session with scan results
      setSession((prev) => ({
        ...prev!,
        currentScan: {
          imagePath: result.image_path,
          segmentationResults: result.components?.segmentation,
          similarCases: result.components?.similar_cases || [],
        },
      }));

      // Generate initial analysis
      const analysisResponse = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: "Please provide an initial analysis of this scan.",
          session_id: session?.sessionId,
          scan_context: {
            image_path: result.image_path,
            segmentation: result.components?.segmentation,
            similar_cases: result.components?.similar_cases,
          },
        }),
      });

      if (analysisResponse.ok) {
        const analysis = await analysisResponse.json();
        const analysisMessage: Message = {
          role: "assistant",
          content: analysis.response || "Scan processed successfully. How can I help you analyze it?",
          timestamp: new Date().toISOString(),
        };

        setSession((prev) => ({
          ...prev!,
          messages: [...prev!.messages.slice(0, -1), analysisMessage],
        }));
      }
    } catch (error: any) {
      const errorMessage: Message = {
        role: "assistant",
        content: `Error processing file: ${error.message}. Please try again.`,
        timestamp: new Date().toISOString(),
      };

      setSession((prev) => ({
        ...prev!,
        messages: [...prev!.messages.slice(0, -1), errorMessage],
      }));
    } finally {
      setUploading(false);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading || !session) return;

    const userMessage: Message = {
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };

    setSession((prev) => ({
      ...prev!,
      messages: [...prev!.messages, userMessage],
    }));

    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: input,
          session_id: session.sessionId,
          scan_context: session.currentScan
            ? {
                image_path: session.currentScan.imagePath,
                segmentation: session.currentScan.segmentationResults,
                similar_cases: session.currentScan.similarCases,
              }
            : undefined,
          conversation_history: session.messages.slice(-10), // Last 10 messages for context
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.response || "I apologize, but I couldn't generate a response.",
        timestamp: new Date().toISOString(),
      };

      setSession((prev) => ({
        ...prev!,
        messages: [...prev!.messages, assistantMessage],
      }));
    } catch (error: any) {
      const errorMessage: Message = {
        role: "assistant",
        content: `Error: ${error.message}. Please try again.`,
        timestamp: new Date().toISOString(),
      };

      setSession((prev) => ({
        ...prev!,
        messages: [...prev!.messages, errorMessage],
      }));
    } finally {
      setIsLoading(false);
    }
  };

  if (!session) {
    return <div className="text-center p-8">Initializing chat...</div>;
  }

  return (
    <div className="flex flex-col h-[calc(100vh-200px)] max-h-[800px]">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4 mb-4">
        {session.messages.map((msg, idx) => (
          <div
            key={idx}
            className={classNames("flex", {
              "justify-end": msg.role === "user",
              "justify-start": msg.role === "assistant",
            })}
          >
            <div
              className={classNames(
                "max-w-[80%] rounded-lg px-4 py-3 shadow-sm",
                {
                  "bg-blue-600 text-white": msg.role === "user",
                  "bg-slate-800 text-slate-100": msg.role === "assistant",
                  "bg-slate-700 text-slate-300": msg.role === "system",
                }
              )}
            >
              <div className="whitespace-pre-wrap break-words">{msg.content}</div>
              {msg.timestamp && (
                <div className="text-xs mt-2 opacity-70">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </div>
              )}
              {msg.imagePath && (
                <div className="text-xs mt-1 opacity-70">📎 {msg.imagePath}</div>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-800 text-slate-100 rounded-lg px-4 py-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* File Upload Area */}
      <div className="border-t border-slate-700 p-4 bg-slate-900/50">
        <div className="mb-3">
          <label className="block text-sm text-slate-400 mb-2">
            Upload Medical Image (DICOM/NIfTI)
          </label>
          <input
            ref={fileInputRef}
            type="file"
            accept=".nii,.nii.gz,.dcm,.dicom"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFileUpload(file);
            }}
            className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 file:cursor-pointer bg-slate-800 border border-slate-700 rounded"
            disabled={uploading}
            aria-label="Upload medical image file (DICOM or NIfTI)"
            title="Upload medical image file"
          />
          {uploading && (
            <div className="mt-2 text-sm text-blue-400">Uploading and processing...</div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-slate-700 p-4 bg-slate-900/50">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="Ask a question about the scan or request analysis..."
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            Send
          </button>
        </div>
        {session.currentScan && (
          <div className="mt-2 text-xs text-slate-400">
            💡 Active scan: {session.currentScan.imagePath.split("/").pop()}
          </div>
        )}
      </div>
    </div>
  );
}

