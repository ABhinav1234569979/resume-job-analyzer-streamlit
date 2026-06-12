import os
import re
import tempfile

import fitz
from flask import Flask, render_template, request, session

from resume_validator import validate_resume_with_ollama
from translations import get_text, get_available_languages


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Maximum uploaded PDF size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


SKILL_ALIASES = {
    "Python": ["python"],
    "Java": ["java"],
    "JavaScript": ["javascript", "java script"],
    "TypeScript": ["typescript"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Bootstrap": ["bootstrap"],
    "React": ["react", "react.js", "reactjs"],
    "Angular": ["angular", "angular.js", "angularjs"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi", "fast api"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "Spring Boot": ["spring boot", "springboot"],
    "REST API": ["rest api", "restful api", "restful services"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo db"],
    "SQLite": ["sqlite"],
    "Redis": ["redis"],
    "Git": ["git"],
    "GitHub": ["github"],
    "GitLab": ["gitlab"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Jenkins": ["jenkins"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous deployment"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "Google Cloud": ["google cloud", "google cloud platform", "gcp"],
    "Linux": ["linux"],
    "Terraform": ["terraform"],
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "Artificial Intelligence": ["artificial intelligence"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Postman": ["postman"],
    "Jira": ["jira"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Cybersecurity": ["cybersecurity", "cyber security"],
    "Networking": ["networking"],
    "Data Structures": ["data structures"],
    "Algorithms": ["algorithms"],
    "Object-Oriented Programming": [
        "object-oriented programming",
        "object oriented programming",
        "oop"
    ]
}


def extract_pdf_text(pdf_path):
    """Validate the PDF structure and extract readable text."""

    with open(pdf_path, "rb") as pdf_file:
        header = pdf_file.read(1024)

    if b"%PDF-" not in header:
        raise ValueError(
            "The uploaded file does not contain a valid PDF signature."
        )

    document = fitz.open(pdf_path)

    try:
        if document.needs_pass:
            raise ValueError(
                "The uploaded PDF is password protected."
            )

        if document.page_count == 0:
            raise ValueError(
                "The uploaded PDF contains no pages."
            )

        pages = []

        for page in document:
            pages.append(page.get_text("text"))

        return "\n".join(pages).strip()

    finally:
        document.close()


def contains_term(text, term):
    """Match complete skill names instead of partial word matches."""

    escaped_term = re.escape(term.lower())
    pattern = rf"(?<!\w){escaped_term}(?!\w)"

    return re.search(pattern, text.lower()) is not None


def extract_skills(text):
    """Extract skills from text using the predefined skill database."""

    detected_skills = set()

    for canonical_skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            if contains_term(text, alias):
                detected_skills.add(canonical_skill)
                break

    return detected_skills


def generate_suggestions(score, missing_skills, language="en"):
    """Generate simple resume-improvement suggestions."""

    suggestions = []

    if missing_skills:
        important_missing = ", ".join(sorted(missing_skills)[:6])

        suggestions.append(
            f"{get_text('suggestion_missing', language)}: {important_missing}"
        )

    if score < 50:
        suggestions.append(
            get_text("suggestion_low_score", language)
        )
    elif score < 75:
        suggestions.append(
            get_text("suggestion_medium_score", language)
        )
    else:
        suggestions.append(
            get_text("suggestion_high_score", language)
        )

    suggestions.append(
        get_text("suggestion_metrics", language)
    )

    suggestions.append(
        get_text("suggestion_headings", language)
    )

    return suggestions


def get_score_label(score):
    if score >= 80:
        return "Excellent Match", "success"

    if score >= 65:
        return "Good Match", "primary"

    if score >= 45:
        return "Average Match", "warning"

    return "Needs Improvement", "danger"


@app.route("/", methods=["GET"])
def home():
    # Set language from request or use default
    language = request.args.get("lang", session.get("language", "en"))
    if language not in get_available_languages():
        language = "en"
    session["language"] = language
    
    return render_template(
        "index.html",
        languages=get_available_languages(),
        current_language=language,
        get_text=lambda key: get_text(key, language)
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    language = session.get("language", "en")
    resume_file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not resume_file or not resume_file.filename:
        return render_template(
            "index.html",
            error=get_text("error_no_resume", language),
            languages=get_available_languages(),
            current_language=language,
            get_text=lambda key: get_text(key, language)
        )

    if not resume_file.filename.lower().endswith(".pdf"):
        return render_template(
            "index.html",
            error=get_text("error_invalid_pdf", language),
            languages=get_available_languages(),
            current_language=language,
            get_text=lambda key: get_text(key, language)
        )

    if not job_description:
        return render_template(
            "index.html",
            error=get_text("error_no_jd", language),
            languages=get_available_languages(),
            current_language=language,
            get_text=lambda key: get_text(key, language)
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temporary_file:
            resume_file.save(temporary_file.name)
            temporary_path = temporary_file.name

        resume_text = extract_pdf_text(temporary_path)

        if len(resume_text) < 30:
            return render_template(
                "index.html",
                error=get_text("error_no_text", language),
                languages=get_available_languages(),
                current_language=language,
                get_text=lambda key: get_text(key, language)
            )

        resume_validation = validate_resume_with_ollama(
            resume_text,
            language=language
        )

        confidence_percent = round(
            resume_validation.confidence * 100
        )

        accepted_document_types = {"resume", "cv"}

        is_verified_resume = (
            resume_validation.is_resume
            and resume_validation.document_type
            in accepted_document_types
            and resume_validation.confidence >= 0.70
        )

        if not is_verified_resume:
            readable_type = (
                resume_validation.document_type
                .replace("_", " ")
                .title()
            )

            return render_template(
                "index.html",
                error=(
                    f"{get_text('error_not_resume', language)} "
                    f"{readable_type}. "
                    f"Confidence: {confidence_percent}%. "
                    f"{resume_validation.reason}"
                ),
                languages=get_available_languages(),
                current_language=language,
                get_text=lambda key: get_text(key, language)
            )

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)

        if not job_skills:
            return render_template(
                "index.html",
                error=get_text("error_no_skills", language),
                languages=get_available_languages(),
                current_language=language,
                get_text=lambda key: get_text(key, language)
            )

        matching_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills.difference(resume_skills)

        score = round(
            len(matching_skills) / len(job_skills) * 100
        )

        score_label, score_color = get_score_label(score)

        result = {
            "filename": resume_file.filename,
            "resume_document_type": (
                resume_validation.document_type
            ),
            "resume_confidence": confidence_percent,
            "resume_sections": (
                resume_validation.detected_sections
            ),
            "score": score,
            "score_label": score_label,
            "score_color": score_color,
            "matching_skills": sorted(matching_skills),
            "missing_skills": sorted(missing_skills),
            "resume_skills": sorted(resume_skills),
            "job_skills": sorted(job_skills),
            "matched_count": len(matching_skills),
            "required_count": len(job_skills),
            "suggestions": generate_suggestions(score, missing_skills, language)
        }

        return render_template(
            "index.html",
            result=result,
            languages=get_available_languages(),
            current_language=language,
            get_text=lambda key: get_text(key, language)
        )

    except Exception as error:
        return render_template(
            "index.html",
            error=f"{get_text('analysis_failed', language)} {error}",
            languages=get_available_languages(),
            current_language=language,
            get_text=lambda key: get_text(key, language)
        )

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@app.errorhandler(413)
def file_too_large(error):
    language = session.get("language", "en")
    return render_template(
        "index.html",
        error=get_text("error_file_too_large", language),
        languages=get_available_languages(),
        current_language=language,
        get_text=lambda key: get_text(key, language)
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
