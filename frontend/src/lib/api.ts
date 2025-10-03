const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function apiStatus() {
  const res = await fetch(`${API_URL}/status`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function apiProcessByPath(imagePath: string) {
  const res = await fetch(`${API_URL}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_path: imagePath })
  });
  if (!res.ok) throw new Error('Process by path failed');
  return res.json();
}

export async function apiProcessUpload(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_URL}/process-upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}


