import os
import re
import tempfile

import fitz
from flask import Flask, render_template, request

from resume_validator import validate_resume_with_ollama


app = Flask(__name__)

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


def generate_suggestions(score, missing_skills):
    """Generate simple resume-improvement suggestions."""

    suggestions = []

    if missing_skills:
        important_missing = ", ".join(sorted(missing_skills)[:6])

        suggestions.append(
            f"The job description emphasizes {important_missing}. "
            "Add these skills only if you genuinely have experience using them."
        )

    if score < 50:
        suggestions.append(
            "Create or highlight projects that demonstrate the main technologies "
            "required by the position."
        )
    elif score < 75:
        suggestions.append(
            "Strengthen your project and experience descriptions using relevant "
            "keywords from the job description."
        )
    else:
        suggestions.append(
            "Your resume has strong skill coverage. Focus on measurable "
            "achievements and role-specific experience."
        )

    suggestions.append(
        "Use measurable achievements, such as performance improvements, "
        "users served, time saved, or accuracy achieved."
    )

    suggestions.append(
        "Use standard resume headings such as Skills, Experience, Education, "
        "and Projects to improve readability."
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
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not resume_file or not resume_file.filename:
        return render_template(
            "index.html",
            error="Please upload your resume as a PDF."
        )

    if not resume_file.filename.lower().endswith(".pdf"):
        return render_template(
            "index.html",
            error="Only PDF resume files are supported."
        )

    if not job_description:
        return render_template(
            "index.html",
            error="Please paste the job description."
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
                error=(
                    "Very little text could be extracted from this PDF. "
                    "Please upload a text-based resume PDF."
                )
            )

        resume_validation = validate_resume_with_ollama(
            resume_text
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
                    "The uploaded PDF does not appear to be a "
                    "genuine resume or CV. "
                    f"Detected document type: {readable_type}. "
                    f"Confidence: {confidence_percent}%. "
                    f"{resume_validation.reason}"
                )
            )

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)

        if not job_skills:
            return render_template(
                "index.html",
                error=(
                    "No recognizable technical skills were found in the job "
                    "description. Try using a more detailed description."
                )
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
            "suggestions": generate_suggestions(score, missing_skills)
        }

        return render_template("index.html", result=result)

    except Exception as error:
        return render_template(
            "index.html",
            error=f"Unable to analyze the resume: {error}"
        )

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="The PDF is too large. Please upload a file smaller than 5 MB."
    ), 413


if __name__ == "__main__":
    app.run(debug=True)
