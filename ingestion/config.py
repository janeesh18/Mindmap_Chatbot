import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env first, fall back to .env.example if .env doesn't exist
_here = Path(__file__).parent
if (_here / ".env").exists():
    load_dotenv(_here / ".env")
else:
    load_dotenv(_here / ".env.example")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", r"C:\Users\janee\OneDrive\文档\chatboit\Sales Collateral"))

# ── OpenAI ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
EMBEDDING_BATCH = 100        # texts per embeddings API call
VLM_MODEL       = "gpt-4o"
IMAGE_DPI       = 150        # resolution for PDF→image render

# ── Qdrant ─────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "mindmap_sales_collateral"

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS    = 380    # ≈ 500 tokens
CHUNK_OVERLAP_WORDS = 60     # ≈ 80 tokens
MIN_CHUNK_WORDS     = 30
ONEPAGER_THRESHOLD  = 400    # docs below this word count stay as a single chunk
UPSERT_BATCH        = 50     # chunks per Qdrant upsert call

# ── File routing ───────────────────────────────────────────────────────────

# Fully image-rendered PDFs — send every page through GPT-4o vision
VLM_REQUIRED_FILES = {
    "19 Billing Reconciliation for Home Care.pdf",
    "20 Cash Reciepts Posting.pdf",
    "21 Claims Ageing Report.pdf",
    "22 Caregiver Onboarding Process.pdf",
    "23 Patient Access Automation.pdf",
    "24 Patient Admissions Process Automation.pdf",
    "25 Appointment Scheduling.pdf",
    "26 Insurance Claims Pre Authorization.pdf",
    "28 Cash Applications.pdf",
    "UAE Bank - RPA NER.pdf",
    "Digital Automation CoE One Slider for RPA.pdf",
    "MindMap AI 1 Slider.pdf",
    "MindMap AWS and GCP Deck.pdf",
    "MindMap Digital 3 pager brief.pdf",
    "MindMap Fractional CFO Advisory.pdf",
    "MindMap Sharepoint Capabilites.pdf",
}

# Image-heavy PDFs whose PPTX twin is better — skip PDF, PPTX will be picked up
SKIP_PDF_USE_PPTX = {
    "MindMap Digital New Deck v5 for gitex.pdf",
    "MindMap Digital HCP DT.pdf",
    "Chatbot Presentation.pdf",
    "Transform Banking.pdf",
    "Transform Origniations.pdf",
    "MindMap Mortgages Pack.pdf",
    "AE for BFSI_V1.0.pdf",
    "MM Trade Processing & Settlement - Case Study.pdf",
}

# Exact duplicates — keep only the canonical version
SKIP_DUPLICATES = {
    "MindMap Digital New Deck v5 for gitex-MMDL_HY_007.pdf",   # dup of gitex PPTX
    "09 Faster Diagnostics for Medical Diagnosis center (1).pdf",
    "Piramal Pharma Case Study.pdf",
    "Cheques Data Extraction Case Study.pdf",
}

# No-PPTX-twin image PDFs — add to VLM so they're not silently dropped
VLM_REQUIRED_FILES.add("MindMap Digital - the Art of Digital Transformation.pdf")

# Entire folders to exclude
SKIP_FOLDERS = {"Videos and Demos"}

# Extensions with no indexable text
SKIP_EXTENSIONS = {".mp4", ".mov", ".gif", ".jpg", ".jpeg", ".png", ".avi", ".xlsx"}

# ── Client name mapping ────────────────────────────────────────────────────
# Maps lowercase keywords found in file names / folder paths → canonical client name
CLIENT_NAME_MAP = {
    "umc":          "United Methodist Communities",
    "parker":       "Parker",
    "fellowship":   "Fellowship Village",
    "ingleside":    "Ingleside",
    "archcare":     "Archcare",
    "intas":        "Intas Pharmaceuticals",
    "piramal":      "Piramal Pharma",
    "wio":          "Wio Bank",
    "uae bank":     "UAE Bank",
    "kotak":        "Kotak",
    "authbridge":   "Authbridge",
    "nga":          "NGA HR",
    "zurich":       "Zurich",
    "ddc":          "TheDDCGroup",
    "bluetide":     "BlueTide",
    "clevercruit":  "CleverCruit",
}

# ── Taxonomy ───────────────────────────────────────────────────────────────
FOLDER_TO_DOCTYPE = {
    "Case Studies":                                     "case_study",
    "new_case_Study":                                   "case_study",
    "HeatMaps":                                         "heatmap",
    "Different Types of Collateral":                    "capability_deck",
    "Client Specific Material which can be referenced": "proposal_client",
    "Samples and Examples":                             "assessment_sample",
    "Vertical Wise":                                    "industry_pack",
    "SDD":                                              "solution_design",
}

FOLDER_TO_VERTICAL = {
    "BFSI":                     ["BFSI"],
    "Healthcare":               ["Healthcare"],
    "F&A":                      ["FA"],
    "HR":                       ["HR"],
    "IT":                       ["IT"],
    "Aviation":                 ["Aviation"],
    "MFG":                      ["Manufacturing"],
    "SCM":                      ["SCM"],
    "Telecom":                  ["Telecom"],
    "Government":               ["Government"],
    "Retail":                   ["Retail"],
    "Education":                ["Education"],
    "Contact Centers":          ["Contact Centers"],
    "Transport and Logistics":  ["Logistics"],
    "SAP":                      ["SAP"],
    "Legal":                    ["Legal"],
    "Utilities":                ["Utilities"],
    "Non Digital":              ["General"],
}
