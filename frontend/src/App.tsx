import { useMemo, useState } from 'react'
import { Loader2, Upload } from 'lucide-react'
import { analyzeResume, sendEmail } from './api'
import ResultsDashboard from './components/ResultsDashboard'
import type { CandidateCard } from './types'

function extractTextFromFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const result = reader.result
        if (typeof result !== 'string') {
          resolve('')
          return
        }
        resolve(result)
      } catch (err) {
        reject(err)
      }
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })
}

export default function App() {
  const [jdText, setJdText] = useState('')
  const [resumeText, setResumeText] = useState('')
  const [resumeFile, setResumeFile] = useState<File | null>(null)
  const [temperature, setTemperature] = useState(0.4)
  const [loading, setLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [remaining, setRemaining] = useState<number | null>(null)
  const [card, setCard] = useState<CandidateCard | null>(null)

  const summaryLabel = useMemo(() => {
    if (temperature < 0.35) return 'Professional / formal'
    if (temperature > 0.65) return 'Warm / startup'
    return 'Balanced / tailored'
  }, [temperature])

  async function handleFile(file: File | null) {
    setResumeFile(file)
    if (!file) return
    if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
      const text = await extractTextFromFile(file)
      setResumeText(text)
    }
  }

  async function handleAnalyze() {
    setError('')
    setLoading(true)
    try {
      const response = await analyzeResume({
        jd_text: jdText,
        resume_text: resumeText.trim() ? resumeText : undefined,
        resume_file: resumeFile,
        temperature,
      })
      setRemaining(response.request_count_remaining ?? null)
      setCard(response.candidate_cards[0] || null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  async function handleSendEmail() {
    if (!card?.candidate_email) return
    setError('')
    setSending(true)
    try {
      await sendEmail({
        candidate_email: card.candidate_email,
        candidate_name: card.candidate_name,
        job_title: card.job_title,
        subject: card.email_subject,
        body: card.email_body,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8 lg:px-8">
        <div className="mb-8 flex flex-col gap-2">
          <h1 className="text-3xl font-semibold text-white">Recruiter Co-Pilot</h1>
          <p className="text-slate-400">Semantic gap analysis between a JD and a resume, with outreach email drafting.</p>
          <div className="text-sm text-slate-500">Requests remaining for this client: {remaining ?? 2}</div>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5 shadow-2xl shadow-black/10">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Input Stage</h2>
              <div className="text-xs text-slate-400">{summaryLabel}</div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Job Description</label>
                <textarea
                  value={jdText}
                  onChange={(e) => setJdText(e.target.value)}
                  className="min-h-96 w-full rounded-2xl border border-slate-700 bg-slate-950 p-4 text-sm text-slate-100 outline-none placeholder:text-slate-500"
                  placeholder="Paste the JD here..."
                />
              </div>

              <div className="space-y-3">
                <label className="text-sm font-medium text-slate-300">Resume</label>
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  className="min-h-72 w-full rounded-2xl border border-slate-700 bg-slate-950 p-4 text-sm text-slate-100 outline-none placeholder:text-slate-500"
                  placeholder="Paste resume text here..."
                />
                <label className="flex cursor-pointer items-center gap-3 rounded-2xl border border-dashed border-slate-700 bg-slate-950 px-4 py-4 text-sm text-slate-300 transition hover:border-slate-500">
                  <Upload className="h-4 w-4" />
                  <span>{resumeFile ? resumeFile.name : 'Upload resume PDF/TXT'}</span>
                  <input
                    type="file"
                    accept=".pdf,.txt"
                    className="hidden"
                    onChange={(e) => void handleFile(e.target.files?.[0] || null)}
                  />
                </label>

                <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-300">Email Temperature</label>
                    <span className="text-xs text-slate-400">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="mt-2 text-xs text-slate-500">
                    Low: formal and precise. High: energetic and startup-friendly.
                  </div>
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={loading || !jdText.trim() || (!resumeText.trim() && !resumeFile)}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-indigo-500 px-4 py-3 font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                  {loading ? 'Analyzing...' : 'Run Analysis'}
                </button>

                {error ? (
                  <div className="rounded-2xl border border-rose-900 bg-rose-950/60 p-3 text-sm text-rose-200">
                    {error}
                  </div>
                ) : null}
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5 shadow-2xl shadow-black/10">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Analysis Stage</h2>
              <div className="text-xs text-slate-400">Candidate Card</div>
            </div>
            <ResultsDashboard card={card} sending={sending} onSendEmail={handleSendEmail} />
          </section>
        </div>
      </div>
    </div>
  )
}
