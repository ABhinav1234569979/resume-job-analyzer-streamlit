import os
import re
import json
from collections import Counter
from dataclasses import dataclass, field

import fitz
import ollama
import streamlit as st

from translations import get_text, get_available_languages


# Initialize language in session state
if "language" not in st.session_state:
    st.session_state.language = "en"


def get_t(key: str) -> str:
    """Shorthand for getting translated text with current language."""
    return get_text(key, st.session_state.language)


st.set_page_config(
    page_title="ResumeMatch",
    page_icon="📄",
    layout="wide"
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(36, 211, 126, 0.16), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(74, 144, 226, 0.12), transparent 35%),
            #05070a;
        color: #f4f7fb;
    }

    h1, h2, h3, h4 {
        font-weight: 900 !important;
        letter-spacing: -0.03em;
    }

    .main-title {
        font-size: 3.4rem;
        font-weight: 950;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #b7c0cd;
        max-width: 900px;
        margin-bottom: 2rem;
    }

    .result-card {
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.045);
        border-radius: 22px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
        box-shadow: 0 18px 50px rgba(0,0,0,0.25);
    }

    .score-card {
        border: 1px solid rgba(72, 255, 158, 0.25);
        background: linear-gradient(135deg, rgba(72,255,158,0.16), rgba(255,255,255,0.04));
        border-radius: 26px;
        padding: 1.4rem;
        margin: 1rem 0;
    }

    .small-muted {
        color: #aab4c2;
        font-size: 0.95rem;
    }

    .chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        margin: 0.2rem;
        border-radius: 999px;
        background: rgba(72,255,158,0.13);
        border: 1px solid rgba(72,255,158,0.25);
        color: #eafff2;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .missing-chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        margin: 0.2rem;
        border-radius: 999px;
        background: rgba(255, 99, 99, 0.13);
        border: 1px solid rgba(255, 99, 99, 0.25);
        color: #ffe9e9;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #49f08f, #27c46c);
        color: #061008;
        border: none;
        border-radius: 16px;
        padding: 0.85rem 1.4rem;
        font-weight: 900;
        font-size: 1rem;
    }

    .stTextArea textarea, .stFileUploader section {
        background-color: rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


SKILL_ALIASES = {
    "Python": ["python", "python developer", "python programming"],
    "Java": ["java", "core java"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node", "node.js", "nodejs"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi"],
    "Streamlit": ["streamlit"],
    "SQL": ["sql", "mysql", "postgresql", "sqlite"],
    "MongoDB": ["mongodb", "mongo"],
    "Git": ["git", "github", "gitlab"],
    "Docker": ["docker", "containerization"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "NLP": ["nlp", "natural language processing"],
    "REST API": ["rest api", "restful api", "api integration"],
    "DSA": ["dsa", "data structures", "algorithms"],
    "OOP": ["oop", "object oriented programming"],
    "Linux": ["linux", "ubuntu"],
    "Excel": ["excel", "microsoft excel"],
    "Communication": ["communication", "verbal communication", "written communication"],
    "Leadership": ["leadership", "team leadership"],
    "Problem Solving": ["problem solving", "analytical thinking"],
    "Documentation": ["documentation", "technical documentation"],
    "Project Management": ["project management"],
    "HR": ["human resources", "hr"],
    "Recruitment": ["recruitment", "talent acquisition"],
    "Compliance": ["compliance"],
    "Training": ["training", "learning and development"],
}


@dataclass
class ResumeValidation:
    is_resume: bool
    confidence: int
    detected_type: str
    reason: str
    validation_source: str


@dataclass
class AIResumeEvaluation:
    overall_fit_score: int = 0
    resume_quality_score: int = 0
    job_alignment_score: int = 0
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    content_improvements: list[str] = field(default_factory=list)
    rewrite_suggestions: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)


def get_ollama_key() -> str:
    try:
        secret_key = st.secrets.get("OLLAMA_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("OLLAMA_API_KEY", "")


def get_ollama_model() -> str:
    try:
        secret_model = st.secrets.get("OLLAMA_MODEL", "")
    except Exception:
        secret_model = ""
    return secret_model or os.getenv("OLLAMA_MODEL", "nemotron-3-nano:30b")


def get_ollama_client():
    api_key = get_ollama_key()
    if not api_key:
        return None
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )


def safe_json_loads(raw_text: str) -> dict:
    text = raw_text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def extract_pdf_text(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    parts = []
    for page in doc:
        parts.append(page.get_text())

    doc.close()
    return "\n".join(parts).strip()


def detect_skills(text: str) -> set[str]:
    text_lower = text.lower()
    detected = set()

    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower()
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias_lower) + r"(?![a-zA-Z0-9])"
            if re.search(pattern, text_lower):
                detected.add(skill)
                break

    return detected


def fast_resume_validation(text: str) -> ResumeValidation:
    text_lower = text.lower()

    resume_markers = [
        "experience", "education", "skills", "projects", "certifications",
        "internship", "work experience", "professional experience",
        "summary", "objective", "resume", "curriculum vitae", "cv",
        "linkedin", "github", "portfolio"
    ]

    non_resume_markers = [
        "abstract", "chapter", "bibliography", "references", "table of contents",
        "invoice", "receipt", "purchase order", "research paper", "journal",
        "news article", "novel"
    ]

    email_found = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
    phone_found = bool(re.search(r"(\+?\d[\d\s\-()]{8,}\d)", text))

    resume_hits = sum(1 for marker in resume_markers if marker in text_lower)
    non_resume_hits = sum(1 for marker in non_resume_markers if marker in text_lower)

    word_count = len(re.findall(r"\w+", text))

    score = resume_hits
    if email_found:
        score += 2
    if phone_found:
        score += 1
    if word_count >= 120:
        score += 1

    if non_resume_hits >= 3 and score < 7:
        return ResumeValidation(
            is_resume=False,
            confidence=75,
            detected_type="Non-resume document",
            reason="The document contains more article/report markers than resume markers.",
            validation_source="Python fast validator"
        )

    if score >= 5:
        return ResumeValidation(
            is_resume=True,
            confidence=min(95, 55 + score * 6),
            detected_type="Resume/CV",
            reason="The document contains common resume sections and contact/profile indicators.",
            validation_source="Python fast validator"
        )

    return ResumeValidation(
        is_resume=False,
        confidence=50,
        detected_type="Uncertain",
        reason="The document does not contain enough resume structure indicators.",
        validation_source="Python fast validator"
    )


def get_language_name(lang_code: str) -> str:
    """Get full language name from language code."""
    language_names = {
        "en": "English",
        "te": "Telugu",
        "hi": "Hindi",
        "ta": "Tamil"
    }
    return language_names.get(lang_code, "English")


def validate_resume_with_ollama_cloud(text: str, force_cloud_ai: bool = True, language: str = "en") -> ResumeValidation:
    api_key = get_ollama_key()

    if not force_cloud_ai:
        fast_result = fast_resume_validation(text)
        if fast_result.confidence >= 75:
            return fast_result

    if not api_key:
        fallback = fast_resume_validation(text)
        fallback.reason += " Ollama API key is missing, so cloud validation could not run."
        fallback.validation_source = "Python fallback because Ollama API key is missing"
        return fallback

    client = get_ollama_client()
    model = get_ollama_model()

    sample = text[:5000]
    lang_name = get_language_name(language)

    prompt = f"""
You are a strict document classifier.

Determine whether the uploaded PDF text is genuinely a resume/CV.

IMPORTANT: Respond in {lang_name} language ONLY.

Return JSON only with this exact schema:
{{
  "is_resume": true,
  "confidence": 0,
  "detected_type": "Resume/CV or Non-resume document or Uncertain",
  "reason": "short reason in {lang_name}"
}}

Rules:
- A resume/CV usually has candidate profile, skills, education, projects, work experience, certifications, or contact details.
- Do not classify a random article, report, book chapter, invoice, brochure, or academic paper as a resume only because it contains technical words.
- Be strict but fair.

PDF text:
{sample}
"""

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only. No markdown."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0}
        )

        try:
            content = response["message"]["content"]
        except Exception:
            content = response.message.content

        data = safe_json_loads(content)

        return ResumeValidation(
            is_resume=bool(data.get("is_resume", False)),
            confidence=int(data.get("confidence", 50)),
            detected_type=str(data.get("detected_type", "Uncertain")),
            reason=str(data.get("reason", "No reason provided.")),
            validation_source=f"Ollama Cloud resume validation using {model}"
        )

    except Exception as exc:
        fallback = fast_resume_validation(text)
        fallback.reason += f" Ollama Cloud validation failed: {exc}"
        fallback.validation_source = "Python fallback because Ollama Cloud failed"
        return fallback


def evaluate_resume_content_with_ollama(
    resume_text: str,
    job_description: str,
    python_ats_score: int,
    matched_skills: list[str],
    missing_skills: list[str],
    language: str = "en"
):
    api_key = get_ollama_key()
    if not api_key:
        return None, get_text("api_key_missing", language)

    client = get_ollama_client()
    model = get_ollama_model()

    resume_sample = resume_text[:7000]
    jd_sample = job_description[:4000]
    lang_name = get_language_name(language)

    prompt = f"""
You are an expert ATS resume reviewer and hiring evaluator.

Your job is NOT only to check whether this is a resume.
You must evaluate the resume content against the provided job description.

IMPORTANT: Respond ENTIRELY in {lang_name} language. All content, including summary, strengths, weaknesses, suggestions, and red flags must be in {lang_name}.

Use the resume text only. Do not invent experience, education, skills, or achievements.

Return valid JSON only with this exact schema:
{{
  "overall_fit_score": 0,
  "resume_quality_score": 0,
  "job_alignment_score": 0,
  "summary": "2-3 sentence direct evaluation in {lang_name}",
  "strengths": ["specific strength 1 in {lang_name}", "specific strength 2 in {lang_name}"],
  "weaknesses": ["specific weakness 1 in {lang_name}", "specific weakness 2 in {lang_name}"],
  "missing_keywords": ["keyword 1", "keyword 2"],
  "content_improvements": ["specific improvement 1 in {lang_name}", "specific improvement 2 in {lang_name}"],
  "rewrite_suggestions": ["rewrite suggestion 1 in {lang_name}", "rewrite suggestion 2 in {lang_name}"],
  "red_flags": ["red flag 1 in {lang_name}", "red flag 2 in {lang_name}"]
}}

Scoring rules:
- overall_fit_score: how suitable the candidate appears for the job, 0 to 100.
- resume_quality_score: clarity, structure, measurable impact, formatting quality, 0 to 100.
- job_alignment_score: how well the resume content matches the job description, 0 to 100.
- Be strict. A weak or generic resume should not get a high score.
- Give practical suggestions, not generic advice.
- Mention missing skills or weak areas only when supported by the job description.

Python keyword ATS score:
{python_ats_score}

Python matched skills:
{matched_skills}

Python missing skills:
{missing_skills}

Job description:
{jd_sample}

Resume text:
{resume_sample}
"""

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only. No markdown. No extra commentary."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1}
        )

        try:
            content = response["message"]["content"]
        except Exception:
            content = response.message.content

        data = safe_json_loads(content)

        evaluation = AIResumeEvaluation(
            overall_fit_score=int(data.get("overall_fit_score", 0)),
            resume_quality_score=int(data.get("resume_quality_score", 0)),
            job_alignment_score=int(data.get("job_alignment_score", 0)),
            summary=str(data.get("summary", "")),
            strengths=list(data.get("strengths", [])),
            weaknesses=list(data.get("weaknesses", [])),
            missing_keywords=list(data.get("missing_keywords", [])),
            content_improvements=list(data.get("content_improvements", [])),
            rewrite_suggestions=list(data.get("rewrite_suggestions", [])),
            red_flags=list(data.get("red_flags", [])),
        )

        return evaluation, None

    except Exception as exc:
        return None, f"Ollama resume content evaluation failed: {exc}"


def calculate_keyword_score(resume_text: str, job_description: str):
    resume_skills = detect_skills(resume_text)
    job_skills = detect_skills(job_description)

    if not job_skills:
        return 0, [], [], sorted(resume_skills), []

    matched = sorted(resume_skills.intersection(job_skills))
    missing = sorted(job_skills.difference(resume_skills))

    score = round((len(matched) / len(job_skills)) * 100)

    suggestions = [
        f"Add or strengthen evidence for {skill}." for skill in missing[:8]
    ]

    return score, matched, missing, sorted(resume_skills), suggestions


def render_chips(items, missing=False):
    if not items:
        st.markdown(f'<p class="small-muted">{get_t("none_detected")}</p>', unsafe_allow_html=True)
        return

    css_class = "missing-chip" if missing else "chip"
    html = " ".join([f'<span class="{css_class}">{item}</span>' for item in items])
    st.markdown(html, unsafe_allow_html=True)


def render_list(items):
    if not items:
        st.write(get_t("no_major_points"))
        return

    for item in items:
        st.markdown(f"- {item}")


# Language selector at the top
col1, col2, col3 = st.columns([2, 1, 1])
with col3:
    languages = get_available_languages()
    selected_lang = st.selectbox(
        "🌐 Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        key="lang_selector",
        label_visibility="collapsed"
    )
    st.session_state.language = selected_lang

st.markdown(f'<div class="main-title">{get_t("main_title")}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">{get_t("subtitle")}</div>',
    unsafe_allow_html=True
)

api_status = get_t("configured") if get_ollama_key() else get_t("missing")
st.info(f"{get_t('ollama_status')}: Model `{get_ollama_model()}` | {get_t('api_key')}: `{api_status}`")

left, right = st.columns([1, 1.35], gap="large")

with left:
    st.subheader(get_t("upload_resume"))
    uploaded_file = st.file_uploader(
        get_t("choose_resume"),
        type=["pdf"]
    )

    st.caption(
        get_t("privacy_note")
    )

    force_ollama_validation = st.checkbox(
        get_t("force_ollama"),
        value=True
    )

    enable_ai_evaluation = st.checkbox(
        get_t("enable_ai"),
        value=True
    )

with right:
    st.subheader(get_t("paste_jd"))
    job_description = st.text_area(
        get_t("job_description"),
        height=260,
        placeholder=get_t("jd_placeholder")
    )

compare_clicked = st.button(get_t("compare_btn"))

if compare_clicked:
    if not uploaded_file:
        st.error(get_t("error_no_resume"))
        st.stop()

    if not job_description.strip():
        st.error(get_t("error_no_jd"))
        st.stop()

    with st.spinner(get_t("extracting")):
        resume_text = extract_pdf_text(uploaded_file)

    if not resume_text:
        st.error(get_t("error_no_text"))
        st.stop()

    with st.spinner(get_t("validating")):
        resume_validation = validate_resume_with_ollama_cloud(
            resume_text,
            force_cloud_ai=force_ollama_validation,
            language=st.session_state.language
        )

    st.info(
        f"{get_t('validation_method')}: {resume_validation.validation_source} | "
        f"{get_t('detected_type')}: {resume_validation.detected_type} | "
        f"{get_t('confidence')}: {resume_validation.confidence}%"
    )

    if not resume_validation.is_resume:
        st.error(get_t("error_not_resume"))
        st.warning(resume_validation.reason)
        st.stop()

    with st.spinner(get_t("calculating_ats")):
        ats_score, matched_skills, missing_skills, resume_skills, suggestions = calculate_keyword_score(
            resume_text,
            job_description
        )

    with st.expander(get_t("view_ats_details"), expanded=False):
        st.caption(
            get_t("ats_baseline")
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(get_t("keyword_ats_score"), f"{ats_score}%")
        c2.metric(get_t("matched_skills"), len(matched_skills))
        c3.metric(get_t("missing_skills"), len(missing_skills))

        st.markdown(f"### {get_t('matching_skills_title')}")
        render_chips(matched_skills)

        st.markdown(f"### {get_t('missing_skills_title')}")
        render_chips(missing_skills, missing=True)

    if enable_ai_evaluation:
        st.markdown("---")
        st.subheader(get_t("ollama_ai_evaluation"))

        with st.spinner(get_t("evaluating")):
            ai_eval, ai_error = evaluate_resume_content_with_ollama(
                resume_text=resume_text,
                job_description=job_description,
                python_ats_score=ats_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                language=st.session_state.language
            )

        if ai_error:
            st.error(ai_error)
            st.stop()

        st.success(f"{get_t('ollama_evaluated')} `{get_ollama_model()}`.")

        m1, m2, m3 = st.columns(3)
        m1.metric(get_t("ai_overall_fit"), f"{ai_eval.overall_fit_score}%")
        m2.metric(get_t("resume_quality"), f"{ai_eval.resume_quality_score}%")
        m3.metric(get_t("job_alignment"), f"{ai_eval.job_alignment_score}%")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### {get_t('ai_summary')}")
        st.write(ai_eval.summary)
        st.markdown('</div>', unsafe_allow_html=True)

        a, b = st.columns(2)

        with a:
            st.markdown(f"### {get_t('strengths')}")
            render_list(ai_eval.strengths)

            st.markdown(f"### {get_t('missing_keywords')}")
            render_list(ai_eval.missing_keywords)

        with b:
            st.markdown(f"### {get_t('weaknesses')}")
            render_list(ai_eval.weaknesses)

            st.markdown(f"### {get_t('red_flags')}")
            render_list(ai_eval.red_flags)

        st.markdown(f"### {get_t('content_improvements')}")
        render_list(ai_eval.content_improvements)

        st.markdown(f"### {get_t('rewrite_suggestions')}")
        render_list(ai_eval.rewrite_suggestions)

    st.markdown("---")
    with st.expander(get_t("view_extracted")):
        st.text_area(get_t("extracted_resume_text"), resume_text, height=300)
