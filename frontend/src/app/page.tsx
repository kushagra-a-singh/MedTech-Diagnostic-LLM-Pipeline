"use client";
import { useCallback, useEffect, useState } from "react";
import classNames from "classnames";
import ChatInterface from "@/components/ChatInterface";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PipelineStatus = {
  pipeline_status?: string;
  components?: Record<string, string>;
  vector_store_stats?: any;
};

export default function Home() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [activeTab, setActiveTab] = useState<"chat" | "legacy">("chat");

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      const json = await res.json();
      setStatus(json);
    } catch (e: any) {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return (
    <div className="space-y-6">
      {/* Hero */}
      <section className="card p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">MedTech Diagnostic LLM Pipeline</h1>
            <p className="text-slate-400 mt-1">
              AI-powered medical imaging diagnostics with conversational interface. 
              MONAI + Swin UNETR segmentation, FAISS retrieval, and medical LLM reporting.
            </p>
          </div>
          <div className="flex gap-2">
            <button className="btn-secondary" onClick={fetchStatus}>
              Refresh Status
            </button>
          </div>
        </div>
      </section>

      {/* Health and Status */}
      <section className="grid md:grid-cols-3 gap-4">
        <div className="card p-4">
          <div className="text-sm text-slate-400">Pipeline</div>
          <div className="mt-1 text-lg font-medium">{status?.pipeline_status || "unknown"}</div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-slate-400">Components</div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
            {status?.components && Object.entries(status.components).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between">
                <span className="text-slate-300">{k}</span>
                <span className={classNames("px-2 py-0.5 rounded", v === 'active' ? 'bg-emerald-600/40' : 'bg-slate-700')}>
                  {v}
                </span>
              </div>
            ))}
            {!status?.components && <div className="text-slate-500">No data</div>}
          </div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-slate-400">Vector Store</div>
          <div className="mt-2 text-sm space-y-1">
            {status?.vector_store_stats ? (
              <>
                <div>Vectors: {status.vector_store_stats.total_vectors}</div>
                <div>Dim: {status.vector_store_stats.dimension}</div>
                <div>Type: {status.vector_store_stats.index_type}</div>
              </>
            ) : (
              <div className="text-slate-500">Not initialized</div>
            )}
          </div>
        </div>
      </section>

      {/* Tab Navigation */}
      <section className="card p-6">
        <div className="flex gap-2 border-b border-slate-700 mb-4">
          <button
            onClick={() => setActiveTab("chat")}
            className={classNames(
              "px-4 py-2 font-medium transition",
              activeTab === "chat"
                ? "text-blue-400 border-b-2 border-blue-400"
                : "text-slate-400 hover:text-slate-300"
            )}
          >
            💬 Chat Interface
          </button>
          <button
            onClick={() => setActiveTab("legacy")}
            className={classNames(
              "px-4 py-2 font-medium transition",
              activeTab === "legacy"
                ? "text-blue-400 border-b-2 border-blue-400"
                : "text-slate-400 hover:text-slate-300"
            )}
          >
            📊 Legacy View
          </button>
        </div>

        {activeTab === "chat" ? (
          <ChatInterface />
        ) : (
          <div className="text-slate-400 text-center py-8">
            Legacy view available. Switch to Chat Interface for the best experience.
          </div>
        )}
      </section>
    </div>
  );
}


