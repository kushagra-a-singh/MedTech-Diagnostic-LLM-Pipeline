"use client";
import { useCallback, useEffect, useState } from "react";
import classNames from "classnames";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PipelineStatus = {
  pipeline_status?: string;
  components?: Record<string, string>;
  vector_store_stats?: any;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imagePath, setImagePath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

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
  }, [fetchStatus]);

  async function submitUpload() {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/process-upload`, { method: "POST", body: form });
      const json = await res.json();
      setResult(json);
    } catch (e: any) {
      setError(e?.message || "Upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function submitPath() {
    if (!imagePath) return;
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_path: imagePath }),
      });
      const json = await res.json();
      setResult(json);
    } catch (e: any) {
      setError(e?.message || "Process failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Hero */}
      <section className="card p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">LLM‑Integrated Medical Imaging Diagnostics</h1>
            <p className="text-slate-400 mt-1">MONAI + Swin UNETR segmentation, FAISS retrieval, MCP‑FHIR context, and medical LLM reporting.</p>
          </div>
          <div className="flex gap-2">
            <button className="btn-secondary" onClick={fetchStatus}>Refresh Status</button>
            <a className="btn-primary" href="https://github.com" target="_blank" rel="noopener noreferrer">Docs</a>
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
                <span className={classNames("px-2 py-0.5 rounded", v === 'active' ? 'bg-emerald-600/40' : 'bg-slate-700')}>{v}</span>
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

      {/* Inputs */}
      <section className="grid md:grid-cols-2 gap-4">
        <div className="card p-6 space-y-4">
          <div>
            <div className="text-sm text-slate-400 mb-2">Upload image file (NIfTI or DICOM)</div>
            <label htmlFor="file-upload" className="block mb-1 text-slate-300">Select file to upload</label>
            <input
              id="file-upload"
              type="file"
              title="Upload medical image file"
              placeholder="Choose a NIfTI or DICOM file"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </div>
          <button disabled={!file || submitting} onClick={submitUpload} className="btn-primary">
            {submitting ? "Processing..." : "Process Uploaded File"}
          </button>
        </div>
        <div className="card p-6 space-y-4">
          <div>
            <div className="text-sm text-slate-400 mb-2">Or process an existing server path</div>
            <input className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" placeholder="e.g., data/inputs/sample.nii.gz" value={imagePath} onChange={(e) => setImagePath(e.target.value)} />
          </div>
          <button disabled={!imagePath || submitting} onClick={submitPath} className="btn-secondary">
            {submitting ? "Processing..." : "Process Path"}
          </button>
        </div>
      </section>

      {error && (
        <div className="card p-4 text-red-300">{error}</div>
      )}

      {/* Results */}
      {result && (
        <section className="grid lg:grid-cols-3 gap-4">
          <div className="card p-6 lg:col-span-2">
            <h2 className="text-lg font-medium">Generated Report</h2>
            <div className="text-slate-400 text-sm mt-1">From configured medical LLM</div>
            <div className="mt-4 whitespace-pre-wrap text-slate-100">
              {result?.components?.medical_report?.report?.report || "No report"}
            </div>
          </div>
          <div className="card p-6">
            <h2 className="text-lg font-medium">Segmentation Metrics</h2>
            <div className="text-slate-400 text-sm mt-1">Basic stats and Dice (if GT found)</div>
            <div className="mt-3 text-sm space-y-1">
              <div>Num classes: {result?.components?.segmentation?.metrics?.num_classes ?? '-'}</div>
              <div>Mean Dice: {result?.components?.segmentation?.metrics?.mean_dice?.toFixed?.(3) ?? '-'}</div>
              <div className="mt-2">Volumes:</div>
              <div className="max-h-40 overflow-auto bg-slate-900/60 rounded p-2 border border-slate-800">
                <pre className="text-xs">{JSON.stringify(result?.components?.segmentation?.metrics?.volumes, null, 2)}</pre>
              </div>
            </div>
          </div>

          <div className="card p-6 lg:col-span-3">
            <h2 className="text-lg font-medium">Similar Cases</h2>
            <div className="text-slate-400 text-sm mt-1">Top retrieved neighbors</div>
            <div className="mt-3 grid md:grid-cols-2 lg:grid-cols-3 gap-3">
              {(result?.components?.similar_cases || []).map((r: any, idx: number) => (
                <div key={idx} className="bg-slate-900/60 border border-slate-800 rounded p-3">
                  <div className="text-sm text-slate-300">Similarity: {r.similarity?.toFixed?.(3)}</div>
                  <div className="text-xs text-slate-400 mt-1 break-all">{r.metadata?.image_path || '-'}</div>
                </div>
              ))}
              {(!result?.components?.similar_cases || result.components.similar_cases.length === 0) && (
                <div className="text-slate-500">No similar cases found</div>
              )}
            </div>
          </div>

          <div className="card p-6 lg:col-span-3">
            <h2 className="text-lg font-medium">Raw Response</h2>
            <div className="mt-3 max-h-96 overflow-auto bg-slate-900/60 p-3 rounded border border-slate-800">
              <pre className="text-xs text-slate-200">{JSON.stringify(result, null, 2)}</pre>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}


