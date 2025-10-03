import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'MedTech Diagnostic Pipeline',
  description: 'LLM-Integrated AI diagnostics for medical imaging',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen">
          <nav className="sticky top-0 z-30 backdrop-blur border-b border-slate-800 bg-slate-900/30">
            <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-6 w-6 rounded bg-brand-500" />
                <span className="font-semibold">MedTech Diagnostic Pipeline</span>
              </div>
              <div className="text-sm text-slate-400 flex items-center gap-4">
                <a className="hover:text-white" href="/">Home</a>
                <a className="hover:text-white" href="/history">History</a>
              </div>
            </div>
          </nav>
          <main className="max-w-6xl mx-auto px-4 py-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}


