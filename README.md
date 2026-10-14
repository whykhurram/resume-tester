# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and
suggests concrete improvements, powered by Google's **Gemini Flash** models.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for targeted keyword matching
- Overall **ATS score (0-100)** plus a 5-part breakdown
  (keywords, formatting, content/impact, structure, readability)
- **Prioritised improvements** (High / Medium / Low), matched & missing keywords
- **Rewrite examples** for weak bullet points
- Automated quick checks (contact info, sections, length, numbers, bullets)
- Shows the extracted text, i.e. what an ATS would actually "see"
- Download the report as JSON

## Quick start (local)

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Get a free API key at <https://aistudio.google.com/apikey>, then either paste it into the
sidebar or set it beforehand:

```bash
export GEMINI_API_KEY="your-key"          # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

You can also create `.streamlit/secrets.toml` (already git-ignored):

```toml
GEMINI_API_KEY = "your-key"
```

## Choosing the model
The sidebar offers `gemini-flash-latest` (default), `gemini-2.5-flash` and
`gemini-2.5-flash-lite`, or type any model name in "Custom model name".
To change the default, set the `GEMINI_MODEL` environment variable.
If you get a "model not found" error, pick another model; Google retires older ones over time.

## Deploy on Streamlit Community Cloud (free)
1. Push this repo to GitHub (public, or private with access granted).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** -> select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings -> Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**. You'll get a public `*.streamlit.app` URL.

> Never commit your API key. If you put your own key in Secrets, anyone using your public app
> spends your quota. For a public app, leave Secrets empty so each visitor enters their own key.

## How scoring works
Gemini is given a fixed rubric and asked for a structured JSON report (temperature 0.2 for
consistency). The score is an AI estimate of ATS-friendliness, **not** the output of any real
ATS (Workday, Greenhouse, Lever, etc. all behave differently). Use it as guidance.

## Limitations
- Scanned/image-only PDFs can't be read (an ATS can't read them either).
- Resume text is sent to the Gemini API; don't upload anything you aren't comfortable sharing.
- Free-tier API keys have rate limits.

## Project structure
```
app.py            # Streamlit app
requirements.txt  # Python dependencies
README.md
```

## License
MIT
