"use client";
import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HistoryPage() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${API_URL}/history`, { cache: 'no-store' });
        const json = await res.json();
        setItems(json.items || []);
      } catch (e: any) {
        setError(e?.message || 'Failed to load history');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">History</h1>
      {loading && <div className="text-slate-400">Loading...</div>}
      {error && <div className="text-red-300">{error}</div>}
      {!loading && !error && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {items.map((it) => (
            <a key={it.id} href={`/history/${it.id}`} className="card p-4 hover:border-brand-500 transition">
              <div className="text-slate-300 font-medium break-all">{it.id}</div>
              <div className="text-slate-500 text-sm mt-1">{it.metadata?.image_path || 'unknown path'}</div>
              <div className="text-slate-500 text-xs mt-2">{it.metadata?.processing_timestamp || ''}</div>
            </a>
          ))}
          {items.length === 0 && <div className="text-slate-500">No history items yet.</div>}
        </div>
      )}
    </div>
  );
}


