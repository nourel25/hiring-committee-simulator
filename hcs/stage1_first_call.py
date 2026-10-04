"""Stage 1: send a CV snippet to the local model and get back a validated Review."""

CV_TEXT = """
Backend Developer, fintech startup (2022-present)
- Built and maintain 6 REST APIs in Python (FastAPI) serving 50,000 daily users
- Designed the PostgreSQL schema for the payments service and led its migration from MySQL
- Cut average API response time by 40% by adding Redis caching
- Containerised services with Docker and deployed them to a single cloud VM

Junior Python Developer, digital agency (2021-2022)
- Fixed bugs and added features to Django websites for client projects
""".strip()

CHECKS = [
    {"id": "TL1", "text": "Has designed (not just used) a database schema for a production system"},
    {"id": "TL2", "text": "Has led or mentored other engineers (people leadership, not technical ownership)"},
    {"id": "TL3", "text": "Has run services on Kubernetes in production"},
]

# The answer key, written before any model run.
EXPECTED = {
    "TL1": {
        "passed": True,
        "evidence": "Designed the PostgreSQL schema for the payments service"
    },
    "TL2": {
        "passed": False, 
        "evidence": None
    },
    "TL3": {
        "passed": False,
        "evidence": "Containerised services with Docker and deployed them to a single cloud VM",
    },
}
