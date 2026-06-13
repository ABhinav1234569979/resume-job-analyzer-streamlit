# ResumeMatch — AI Resume & Job Description Analyzer

ResumeMatch is an AI-powered web application that compares a resume against a job description and provides AI-based resume feedback, job alignment scoring, and keyword-based ATS insights.

The project is designed to help users understand how well a resume matches a specific role, identify missing skills, improve resume content, and receive practical suggestions for better job alignment.

## Live Demo

```text
https://resume-match-abhinav.streamlit.app
```

## Repository

```text
https://github.com/ABhinav1234569979/resume-job-analyzer-streamlit
```

## Features

- Upload a resume in PDF format
- Paste a job description
- Extract readable text from the uploaded PDF
- Validate whether the uploaded document is actually a resume/CV
- Evaluate resume content using Ollama Cloud
- Compare resume content against the job description
- Generate AI-based scores for:
  - Overall Fit
  - Resume Quality
  - Job Alignment
- Identify:
  - Strengths
  - Weaknesses
  - Missing keywords
  - Content improvement areas
  - Rewrite suggestions
  - Possible red flags
- Provide keyword-based ATS details in an expandable section

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Ollama Cloud API
- GitHub
- Streamlit Community Cloud

## How It Works

1. The user uploads a resume PDF.
2. The app extracts text from the PDF using PyMuPDF.
3. The app checks whether the uploaded document appears to be a genuine resume/CV.
4. The user pastes a job description.
5. The app performs a keyword-based ATS comparison.
6. Ollama Cloud evaluates the resume content against the job description.
7. The app displays AI-powered scores, strengths, weaknesses, missing keywords, and improvement suggestions.

## Ollama AI Usage

Ollama Cloud is used for two major tasks:

### 1. Resume Validation

The model checks whether the uploaded PDF is actually a resume or CV, instead of a random article, invoice, report, book chapter, or unrelated document.

### 2. Resume Content Evaluation

The model evaluates the resume against the job description and provides:

- Overall fit score
- Resume quality score
- Job alignment score
- Resume strengths
- Resume weaknesses
- Missing keywords
- Content improvement suggestions
- Rewrite suggestions
- Red flags

## Keyword ATS Baseline

The app also includes a keyword-based ATS baseline.

This section is intentionally placed inside an expandable details panel because the main focus of the app is the Ollama-powered AI resume evaluation. The keyword section is kept for transparency and explainability.

## Project Structure

```text
resume-job-analyzer/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── app.py
├── resume_validator.py
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

The deployed Streamlit version uses:

```text
streamlit_app.py
```

as the main entry point.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/ABhinav1234569979/resume-job-analyzer-streamlit.git
cd resume-job-analyzer-streamlit
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run streamlit_app.py
```

## Environment Variables

The app uses Ollama Cloud for AI validation and resume evaluation.

For local testing on Windows PowerShell:

```powershell
$env:OLLAMA_API_KEY = "your_ollama_api_key"
$env:OLLAMA_MODEL = "nemotron-3-nano:30b"
```

For Streamlit Community Cloud, add the following in:

```text
Manage app → Settings → Secrets
```

```toml
OLLAMA_API_KEY = "your_ollama_api_key"
OLLAMA_MODEL = "nemotron-3-nano:30b"
```

Do not commit API keys to GitHub.

## Deployment

The app is deployed using Streamlit Community Cloud.

Deployment settings:

```text
Repository: ABhinav1234569979/resume-job-analyzer-streamlit
Branch: main
Main file path: streamlit_app.py
```

After pushing changes to GitHub, Streamlit Community Cloud automatically redeploys the app. If the app does not update immediately, reboot it from the Streamlit dashboard.

## Git Workflow

After making changes:

```bash
git status
git add .
git commit -m "Update project"
git push github main
```

## Current Model

The current Ollama Cloud model used by the app is:

```text
nemotron-3-nano:30b
```

This model is used for resume validation and resume-job description evaluation.

## Limitations

- The app works best with text-based PDF resumes.
- Scanned image-only PDFs may not extract correctly.
- Keyword ATS scoring is only a baseline.
- AI feedback depends on the quality of the resume text, job description, and model response.
- The app does not verify whether resume claims are factually true.

## Future Improvements

- Add OCR support for scanned resumes
- Add DOCX resume upload support
- Add downloadable PDF reports
- Add semantic skill extraction
- Add resume rewrite generation
- Add saved analysis history
- Add role-specific scoring templates

## Project Note

This project was developed as an AI-powered resume analysis and job matching application using Streamlit and Ollama Cloud.
