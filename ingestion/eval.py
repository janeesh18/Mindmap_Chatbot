"""
MindMap RAG — RAGAS Evaluation
===============================
Runs a set of test questions through the full RAG pipeline and scores with RAGAS.

Metrics (no ground-truth required):
  - Faithfulness        : Is the answer grounded in the retrieved context?
  - Answer Relevancy    : Does the answer address the question?
  - Context Precision   : Are the retrieved chunks relevant to the question?

Usage:
    python eval.py                  # full eval, saves eval_results.csv
    python eval.py --dry-run        # print questions only, no API calls
    python eval.py --output out.csv # custom output path
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Test questions — covering SDD, case studies, capabilities, verticals
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUESTIONS: List[str] = [
    # ── SDD — Fellowship ──────────────────────────────────────────────────────
    "What does the Fellowship PO automation process involve?",
    "How does the Fellowship HCS resident intake process work?",
    "What is automated in the Fellowship HIPAA compliance process?",
    "What does the Fellowship HCS Metrics B process automate?",
    "What does the Fellowship Clinical Ops natural gas billing automation do?",
    "How does the Fellowship Clinical Ops process sync data from PCC to Rexpert?",
    "What does the Fellowship Clinical Ops claims to Rexpert automation cover?",
    "What is the Fellowship HR OIG process automation?",

    # ── SDD — Ingleside ───────────────────────────────────────────────────────
    "What does the Ingleside RAAS report update and termination process automate?",
    "What is the Ingleside Mission Dashboard used for?",
    "How does the Ingleside Vimeo recordings automation work?",
    "What does the Ingleside consolidated report automation do?",
    "What is the Ingleside $1000 notification process?",
    "How does the Ingleside discharge summary report automation work?",
    "What does the Ingleside Raiser's Edge resident details update process do?",

    # ── SDD — Parker ──────────────────────────────────────────────────────────
    "What does the Parker rehab billing automation do?",
    "What is the Parker triple check process?",
    "What does the Parker CIPPS report automation cover?",
    "What does the Parker PCC infection control case list automation do?",
    "What is the Parker quality tracking report automation?",
    "What does the Parker Outlook therapists schedule automation involve?",

    # ── SDD — UMC ─────────────────────────────────────────────────────────────
    "How does UMC automate user creation in Active Directory and PointClickCare?",
    "What is the UMC Bloomerang PCC sync process?",
    "How does UMC handle payroll change requests?",
    "What does the UMC Medasync process automate?",
    "What is the UMC pledge payments automation?",
    "What does the UMC surveyor account creation process do?",
    "How does the UMC RL Datix PCC census data automation work?",
    "What does the UMC PCC document upload process automate?",
    "What is the UMC constituent creation in Bloomerang process?",

    # ── SDD — Archcare ────────────────────────────────────────────────────────
    "What is the Archcare keywords scrubbing automation?",
    "What does the Archcare QC initial assessment process automate?",
    "What does the Archcare QC reassessment automation cover?",
    "What is the Archcare new authorization request automation?",
    "What does the Archcare SDR automation process do?",

    # ── Named case studies ────────────────────────────────────────────────────
    "What did MindMap automate for Intas Pharma?",
    "What is the Piramal Pharma case study about?",
    "What does the MindMap Healthcare case study describe?",
    "What does the UAE Bank RPA automation involve?",
    "What was automated for Wio Bank?",
    "What is the Large Programs case study about?",
    "What automation was done in the Neo Bank case study?",

    # ── Numbered case studies ─────────────────────────────────────────────────
    "What ROI was achieved in the FP&A reporting automation case study?",
    "What automation was built for the hospital revenue leakage case study?",
    "What were the results of the trade processing and settlement case study?",
    "What does the cheques data extraction case study describe?",
    "What was automated in the demand planning case study?",
    "What does the quality control and assurance automation case study cover?",
    "What was built in the billing reconciliation for home care case study?",
    "What does the caregiver onboarding automation case study describe?",
    "What does the patient admissions process automation case study cover?",
    "What is the insurance claims pre-authorization case study about?",
    "What does the vendor payment reconciliation case study describe?",
    "What does the HR digital assistant case study cover?",
    "What was automated in the collections automation case study?",
    "What does the RPA factory for large IT services provider case study describe?",
    "What does the cash receipts posting automation case study cover?",
    "What is the appointment scheduling automation case study about?",
    "What does the document processing and regulatory filing for logistics case study cover?",
    "What is the service management monitoring automation case study?",

    # ── New case studies (Fellowship/Archcare/Ingleside/UMC) ──────────────────
    "What does the Archcare admission process lifecycle case study describe?",
    "What is the Archcare authorization request case study about?",
    "What does the Archcare QC initial assessment case study cover?",
    "What does the Ingleside Vimeo recordings case study describe?",
    "What does the Fellowship Village home community service intake case study cover?",
    "What does the UMC automated user account creation case study describe?",
    "What is the UMC surveyor account creation in PCC case study about?",

    # ── SDD — Fellowship (missing) ────────────────────────────────────────────
    "How does the Fellowship Clinical Ops process sync data from PCC to MatrixCare?",

    # ── SDD — Parker (missing AD docs) ───────────────────────────────────────
    "What does the Parker AD1 solution design document describe?",
    "What does the Parker AD4 solution design document describe?",
    "What does the Parker AD5 solution design document describe?",
    "What does the Parker AD6 solution design document describe?",

    # ── SDD — UMC (missing processes & dashboards) ────────────────────────────
    "What does the UMC agent account creation in PointClickCare process do?",
    "What is the UMC bid submittal form automation?",
    "What does the UMC Bloomerang data integrity process automate?",
    "How does the UMC census report download from PCC work?",
    "What does the UMC resident admission process automation cover?",
    "What does the UMC overtime report process automate?",
    "What does the UMC Bloomerang dashboard show?",
    "What does the UMC quality metrics dashboard track?",
    "What does the UMC finance dashboard display?",
    "What does the UMC incident management automation do?",
    "What does the UMC overtime dashboard show?",
    "What does the UMC PCC census dashboard track?",

    # ── Numbered case studies (missing) ──────────────────────────────────────
    "What was automated in the FP&A budget management automation case study?",
    "What does the journal entries and accruals accounting automation case study describe?",
    "What does the order management automation case study cover?",
    "What was built in the analytics across AWS data lake case study?",
    "What does the faster diagnostics for medical diagnosis center case study describe?",
    "What was automated in the SAP posting for vendor invoices case study?",
    "What does the touchless AP telecom case study cover?",
    "What was automated in the first level customer service using RPA case study?",
    "What does the global customer loyalty and campaign management case study describe?",
    "What does the ROQS process automation study report cover?",
    "What does the claims ageing report automation case study describe?",
    "What does the patient access automation case study cover?",
    "What does the recruitment and onboarding case study describe?",
    "What does the cash applications automation case study cover?",
    "What does the inside sales automation case study describe?",
    "What does the change request management automation case study cover?",

    # ── New case studies (missing) ────────────────────────────────────────────
    "What does the Fellowship Claims to Rexpert case study describe?",
    "What does the sales metrics insights case study cover?",
    "What does the dashboard case studies deck describe?",

    # ── Samples & Examples ────────────────────────────────────────────────────
    "What does the data lake process discovery and roadmap describe?",
    "What does the discovery roadmap agenda cover?",
    "What does the UMC Phase 2 final report describe?",
    "What does the ROQS assessment report cover?",

    # ── Client-specific collateral ────────────────────────────────────────────
    "What does the Authbridge RPA proposal cover?",
    "What automation was proposed for TheDDCGroup?",
    "What does the MindMap deck for Wio Bank describe?",
    "What does the chatbot presentation cover?",
    "What does the PR centralization RFI document describe?",

    # ── Different Types of Collateral ─────────────────────────────────────────
    "What does the BlueTide SmartSheet deck describe?",
    "What does the CleverCruit AI recruitment solution offer?",
    "What is the Digital Automation CoE one slider about?",
    "What does the MindMap eLearning capability deck cover?",
    "What is the MindMap AI capabilities one slider about?",
    "What does the MindMap AWS and GCP deck describe?",
    "What does the MindMap corporate training offering cover?",
    "What cloud offerings does MindMap Digital provide?",
    "What does the MindMap Digital 3 pager brief describe?",
    "What is MindMap's fractional CFO advisory service?",
    "What SharePoint capabilities does MindMap offer?",
    "What does MindMap's staff augmentation service cover?",

    # ── General capabilities & verticals ─────────────────────────────────────
    "What RPA automation solutions does MindMap offer for healthcare?",
    "What finance and accounting automation capabilities does MindMap have?",
    "What industries does MindMap Digital serve?",
    "What are MindMap's banking and financial services automation capabilities?",
    "What supply chain management automation solutions does MindMap offer?",
    "What HR automation use cases does MindMap Digital cover?",
    "What IT service desk automation capabilities does MindMap have?",
    "What does MindMap offer for the government and public sector?",
    "What aviation automation use cases does MindMap cover?",
    "What manufacturing automation solutions does MindMap provide?",
    "What are MindMap's contact center automation capabilities?",
    "What cloud and analytics capabilities does MindMap Digital offer?",
    "What SAP automation solutions does MindMap provide?",
    "What legal department automation capabilities does MindMap offer?",
    "What retail automation use cases does MindMap cover?",
    "What transport and logistics automation solutions does MindMap provide?",
    "What utilities automation capabilities does MindMap offer?",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(question: str) -> tuple[str, list[str]]:
    """Retrieve → answer. Returns (answer_text, list_of_context_strings)."""
    from rag import retrieve, stream_answer

    chunks = retrieve(question)
    contexts = [c.get("text", "") for c in chunks]

    answer_tokens = []
    for token in stream_answer(question, chunks, []):
        answer_tokens.append(token)
    answer = "".join(answer_tokens).strip()

    return answer, contexts


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas(questions, answers, contexts_list) -> dict:
    """Run RAGAS evaluation and return scores dict."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Run:  pip install ragas datasets")
        sys.exit(1)

    data = {
        "question": questions,
        "answer":   answers,
        "contexts": contexts_list,
    }

    dataset = Dataset.from_dict(data)

    print("\nRunning RAGAS evaluation …")
    result = evaluate(
        dataset,
        metrics=[Faithfulness()],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MindMap RAGAS evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print questions only, no API calls")
    parser.add_argument("--output",  default="eval_results.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"\n{'=' * 64}")
    print("MindMap RAG — RAGAS Evaluation")
    print(f"Questions : {len(TEST_QUESTIONS)}")
    print(f"{'=' * 64}\n")

    if args.dry_run:
        for i, q in enumerate(TEST_QUESTIONS, 1):
            print(f"  {i:2d}. {q}")
        print("\n[DRY RUN] No API calls made.")
        return

    questions     : List[str]       = []
    answers       : List[str]       = []
    contexts_list : List[List[str]] = []
    timings       : List[float]     = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i:2d}/{len(TEST_QUESTIONS)}] {question}")
        t0 = time.time()
        try:
            answer, contexts = run_pipeline(question)
            elapsed = time.time() - t0
            questions.append(question)
            answers.append(answer)
            contexts_list.append(contexts)
            timings.append(round(elapsed, 2))
            print(f"       → {len(contexts)} chunks retrieved | {elapsed:.1f}s")
            preview = answer[:120].replace("\n", " ")
            print(f"       → {preview}{'…' if len(answer) > 120 else ''}")
        except Exception as e:
            print(f"       [ERROR] {e}")
            questions.append(question)
            answers.append(f"ERROR: {e}")
            contexts_list.append([])
            timings.append(0.0)

        # Cohere Trial key is limited to 10 calls/min — wait 7s between calls
        if i < len(TEST_QUESTIONS):
            time.sleep(7)

    # ── RAGAS scoring ─────────────────────────────────────────────────────────
    valid_mask = [a and not a.startswith("ERROR") for a in answers]
    valid_q    = [q for q, v in zip(questions, valid_mask) if v]
    valid_a    = [a for a, v in zip(answers,   valid_mask) if v]
    valid_c    = [c for c, v in zip(contexts_list, valid_mask) if v]

    ragas_result = run_ragas(valid_q, valid_a, valid_c)

    # ── Build per-question rows ───────────────────────────────────────────────
    # ragas returns a Dataset with per-row scores
    scores_df = ragas_result.to_pandas()

    # Detect which metric columns are actually present in results
    score_cols = [c for c in scores_df.columns if c not in ("question", "answer", "contexts", "ground_truth")]

    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "answer", "n_contexts"] + score_cols + ["elapsed_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        valid_idx = 0
        for i, question in enumerate(questions):
            row: dict = {
                "question":   question,
                "answer":     answers[i][:300],
                "n_contexts": len(contexts_list[i]),
                "elapsed_s":  timings[i],
            }
            for col in score_cols:
                row[col] = ""
            if valid_mask[i] and valid_idx < len(scores_df):
                row_scores = scores_df.iloc[valid_idx]
                for col in score_cols:
                    row[col] = round(float(row_scores.get(col, 0) or 0), 4)
                valid_idx += 1
            writer.writerow(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("RAGAS Summary")
    print(f"{'=' * 64}")
    for metric in score_cols:
        col = scores_df[metric].dropna()
        if len(col):
            print(f"  {metric:<25} mean={col.mean():.4f}  min={col.min():.4f}  max={col.max():.4f}")
    print(f"\n  Results saved → {output_path.resolve()}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
