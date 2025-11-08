"use client";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HistoryItemPage({ params }: { params: { id: string } }) {
  const { id } = params;
  const [item, setItem] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${API_URL}/history/${id}`, { cache: 'no-store' });
        const json = await res.json();
        setItem(json);
      } catch (e: any) {
        setError(e?.message || 'Failed to load item');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <a href="/history" className="btn-secondary">← Back</a>
        <h1 className="text-2xl font-semibold">Item: {id}</h1>
      </div>
      {loading && <div className="text-slate-400">Loading...</div>}
      {error && <div className="text-red-300">{error}</div>}
      {item && (
        <div className="grid lg:grid-cols-2 gap-4">
          <div className="card p-4">
            <h2 className="font-medium">Metadata</h2>
            <div className="mt-2 max-h-96 overflow-auto bg-slate-900/60 p-3 rounded border border-slate-800">
              <pre className="text-xs">{JSON.stringify(item.metadata, null, 2)}</pre>
            </div>
          </div>
          <div className="card p-4">
            <h2 className="font-medium">Files</h2>
            <div className="mt-2 text-sm space-y-1 break-all">
              <div>Mask: {item.mask_path || '-'}</div>
              <div>Embeddings: {item.embedding_path || '-'}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


