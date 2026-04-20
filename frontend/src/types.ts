export type CandidateCard = {
  candidate_name?: string | null
  candidate_email?: string | null
  job_title?: string | null
  match_percentage: number
  strengths: string[]
  gaps: string[]
  missing_but_relevant: string[]
  soft_skill_alignment: string[]
  email_subject: string
  email_body: string
  processed: boolean
  email_pending: boolean
  match_breakdown?: Record<string, number>
}

export type AnalysisResponse = {
  candidate_cards: CandidateCard[]
  processed: boolean
  email_pending: boolean
  request_count_remaining?: number | null
  message?: string | null
}

export type SendEmailPayload = {
  candidate_email: string
  candidate_name?: string | null
  job_title?: string | null
  subject?: string | null
  body?: string | null
}
