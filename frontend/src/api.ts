import type { AnalysisResponse, SendEmailPayload } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function clientId(): string {
  const key = 'recruiter_copilot_client_id'
  const existing = localStorage.getItem(key)
  if (existing) return existing
  const generated = crypto.randomUUID()
  localStorage.setItem(key, generated)
  return generated
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init.headers || {}),
      'X-Client-Id': clientId(),
    },
  })

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`
    try {
      const payload = await response.json()
      detail = payload.detail || payload.message || detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }

  return response.json() as Promise<T>
}

export async function analyzeResume(payload: {
  jd_text: string
  resume_text?: string
  resume_file?: File | null
  temperature: number
}): Promise<AnalysisResponse> {
  const form = new FormData()
  form.append('jd_text', payload.jd_text)
  form.append('temperature', String(payload.temperature))
  if (payload.resume_text) form.append('resume_text', payload.resume_text)
  if (payload.resume_file) form.append('resume_file', payload.resume_file)

  return requestJson<AnalysisResponse>('/api/analyze', {
    method: 'POST',
    body: form,
  })
}

export async function sendEmail(payload: SendEmailPayload) {
  return requestJson<{ sent: boolean; provider_result: unknown }>('/api/send-email', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
