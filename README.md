# Recruiter Co-Pilot

## Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Frontend
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## Notes
- Put `OPENAI_API_KEY` and `SENDGRID_API_KEY` in `backend/.env`
- Keep sender/templates/weights in `backend/config.json`
- The frontend stores a client id in localStorage and sends it in `X-Client-Id`
- Analysis requests are capped at 2 per client window to control LLM usage
