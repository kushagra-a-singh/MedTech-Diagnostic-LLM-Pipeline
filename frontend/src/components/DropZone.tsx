"use client";
import { useCallback } from "react";

export default function DropZone({ onFile }: { onFile: (f: File) => void }) {
  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  }, [onFile]);

  const onSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onFile(f);
  }, [onFile]);

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      className="border-2 border-dashed border-slate-700 rounded p-6 text-center bg-slate-900/50"
    >
      <div className="text-slate-300">Drag and drop a file here</div>
      <div className="text-slate-500 text-sm mt-1">or click to browse</div>
      <label htmlFor="file-upload" className="mt-3 block text-slate-400 text-sm">
        Choose a file
      </label>
      <input
        id="file-upload"
        type="file"
        className="mt-3"
        onChange={onSelect}
        title="Select a file to upload"
        placeholder="Select a file"
      />
    </div>
  );
}


