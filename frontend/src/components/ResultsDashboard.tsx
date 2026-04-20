import { AlertCircle, Mail, Sparkles } from 'lucide-react'
import type { CandidateCard } from '../types'

function CircularProgress({ value }: { value: number }) {
  const normalized = Math.max(0, Math.min(100, value))
  const strokeWidth = 10
  const radius = 48
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (normalized / 100) * circumference

  return (
    <div className="relative h-28 w-28">
      <svg className="-rotate-90 h-28 w-28">
        <circle cx="56" cy="56" r={radius} stroke="currentColor" strokeWidth={strokeWidth} fill="transparent" className="text-slate-700" />
        <circle
          cx="56"
          cy="56"
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="text-emerald-400 transition-all"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-semibold text-white">{Math.round(normalized)}%</span>
        <span className="text-xs text-slate-400">match</span>
      </div>
    </div>
  )
}

function BulletList({ title, items, tone }: { title: string; items: string[]; tone: 'good' | 'bad' }) {
  const color = tone === 'good' ? 'text-emerald-300' : 'text-rose-300'
  const dot = tone === 'good' ? 'bg-emerald-400' : 'bg-rose-400'
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
        {tone === 'good' ? <Sparkles className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
        {title}
      </div>
      <ul className="space-y-2">
        {items.slice(0, tone === 'good' ? 3 : 2).map((item, idx) => (
          <li key={idx} className={`flex gap-3 text-sm ${color}`}>
            <span className={`mt-2 h-2 w-2 rounded-full ${dot}`} />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function ResultsDashboard({
  card,
  onSendEmail,
  sending,
}: {
  card: CandidateCard | null
  onSendEmail: () => void
  sending: boolean
}) {
  if (!card) {
    return (
      <div className="rounded-3xl border border-dashed border-slate-800 bg-slate-950 p-6 text-slate-400">
        Run an analysis to see the candidate card here.
      </div>
    )
  }

  return (
    <div className="space-y-5 rounded-3xl border border-slate-800 bg-slate-950/70 p-5 shadow-2xl shadow-black/20">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-5">
          <CircularProgress value={card.match_percentage} />
          <div>
            <div className="text-xl font-semibold text-white">{card.candidate_name || 'Unnamed Candidate'}</div>
            <div className="text-sm text-slate-400">{card.candidate_email || 'No email extracted'}</div>
            <div className="mt-2 text-sm text-slate-300">{card.job_title || 'Role summary unavailable'}</div>
          </div>
        </div>

        <button
          onClick={onSendEmail}
          disabled={sending || !card.candidate_email}
          className="inline-flex items-center gap-2 rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Mail className="h-4 w-4" />
          {sending ? 'Sending...' : 'Send Outreach'}
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <BulletList title="Strengths" items={card.strengths || []} tone="good" />
        <BulletList title="Gaps / Missing Skills" items={card.gaps || []} tone="bad" />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
          <div className="mb-2 text-sm font-semibold text-slate-100">Missing but Relevant</div>
          <ul className="space-y-2 text-sm text-slate-300">
            {(card.missing_but_relevant || []).slice(0, 4).map((item, idx) => (
              <li key={idx}>• {item}</li>
            ))}
          </ul>
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
          <div className="mb-2 text-sm font-semibold text-slate-100">Soft Skill Alignment</div>
          <ul className="space-y-2 text-sm text-slate-300">
            {(card.soft_skill_alignment || []).slice(0, 4).map((item, idx) => (
              <li key={idx}>• {item}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
        <div className="mb-2 text-sm font-semibold text-slate-100">Draft Email</div>
        <div className="mb-3 text-xs uppercase tracking-wider text-slate-400">{card.email_subject}</div>
        <pre className="whitespace-pre-wrap text-sm leading-6 text-slate-200">{card.email_body}</pre>
      </div>
    </div>
  )
}
