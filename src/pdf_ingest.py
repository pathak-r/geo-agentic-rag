"""
PDF Ingestion Pipeline
Parses well reports and drilling reports, chunks them for RAG.
"""
import os
import re
import pdfplumber
from typing import List, Dict
from src.config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text


def extract_metadata_from_filename(filename: str) -> Dict:
    """
    Extract well name and date from PDF filename.
    Examples:
        15_9_F_11_2013_03_08.pdf -> well=15/9-F-11, date=2013-03-08
        15_9_19_A_1997_07_30.pdf -> well=15/9-19A, date=1997-07-30
        F12_COMPLETION_REPORT_1.PDF -> well=F-12, type=completion_report
    """
    metadata = {"source_file": filename}

    low = filename.lower()
    if low.endswith(".pdf.download"):
        name = filename[: -len(".pdf.download")]
    elif low.endswith(".pdf"):
        name = filename[: -4]
    else:
        name = filename

    # Try to match daily drilling report pattern: 15_9_F_11_2013_03_08
    daily_match = re.match(
        r"15_9_F[_-]?(\d+)_(\d{4})_(\d{2})_(\d{2})", name
    )
    if daily_match:
        well_num = daily_match.group(1)
        year, month, day = daily_match.groups()[1:]
        metadata["well_name"] = f"15/9-F-{well_num}"
        metadata["date"] = f"{year}-{month}-{day}"
        metadata["doc_type"] = "daily_drilling_report"
        return metadata

    # Sidetrack daily: 15_9_F_1_C_2014_02_22
    sidetrack_daily = re.match(
        r"15_9_F_(\d+)_([A-Z])_(\d{4})_(\d{2})_(\d{2})", name
    )
    if sidetrack_daily:
        well_num, sidetrack = sidetrack_daily.group(1), sidetrack_daily.group(2)
        y, mo, d = sidetrack_daily.groups()[2:]
        metadata["well_name"] = f"15/9-F-{well_num} {sidetrack}"
        metadata["date"] = f"{y}-{mo}-{d}"
        metadata["doc_type"] = "daily_drilling_report"
        return metadata

    # Try exploration well pattern: 15_9_19_A_1997_07_30
    expl_match = re.match(
        r"15_9_19_([A-Z])_(\d{4})_(\d{2})_(\d{2})", name
    )
    if expl_match:
        sidetrack = expl_match.group(1)
        year, month, day = expl_match.groups()[1:]
        metadata["well_name"] = f"15/9-19{sidetrack}"
        metadata["date"] = f"{year}-{month}-{day}"
        metadata["doc_type"] = "daily_drilling_report"
        return metadata

    # Completion reports: F11_, F4_, F15D_, etc.
    comp_match = re.match(r"F(\d+)([A-Z])?_COMPLETION", name, re.IGNORECASE)
    if comp_match:
        well_num = comp_match.group(1)
        suffix = (comp_match.group(2) or "").upper()
        if suffix:
            metadata["well_name"] = f"15/9-F-{well_num} {suffix}"
        else:
            metadata["well_name"] = f"15/9-F-{well_num}"
        metadata["doc_type"] = "completion_report"
        return metadata

    # FWR (Final Well Report) pattern
    fwr_match = re.match(r"FWR_Completion_F(\d+)", name, re.IGNORECASE)
    if fwr_match:
        well_num = fwr_match.group(1)
        metadata["well_name"] = f"15/9-F-{well_num}"
        metadata["doc_type"] = "final_well_report"
        return metadata

    # 15-9-F-1-C pattern
    fc_match = re.match(r"15-9-F-(\d+)-([A-Z]+)", name, re.IGNORECASE)
    if fc_match:
        well_num = fc_match.group(1)
        sidetrack = fc_match.group(2)
        metadata["well_name"] = f"15/9-F-{well_num} {sidetrack}"
        metadata["doc_type"] = "completion_report"
        return metadata

    # Fallback
    metadata["well_name"] = "Unknown"
    metadata["doc_type"] = "unknown"
    return metadata


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    # Split on paragraph boundaries first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from end of current chunk
                words = current_chunk.split()
                overlap_words = words[-overlap // 4:] if len(words) > overlap // 4 else words
                current_chunk = " ".join(overlap_words) + "\n\n" + para + "\n\n"
            else:
                # Single paragraph exceeds chunk size — force split
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current_chunk = ""

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_all_pdfs(pdf_dir: str = None) -> List[Dict]:
    """
    Process all PDFs in the directory.
    Returns list of {text, metadata} dicts ready for embedding.
    """
    pdf_dir = pdf_dir or PDF_DIR
    documents = []

    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return documents

    pdf_files = [
        f
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf") or f.lower().endswith(".pdf.download")
    ]

    print(f"Found {len(pdf_files)} PDF files to process")

    for filename in sorted(pdf_files):
        filepath = os.path.join(pdf_dir, filename)
        if not os.path.isfile(filepath):
            print(f"  Skip (not a file — remove incomplete .download folder if needed): {filename}")
            continue
        print(f"  Processing: {filename}")

        try:
            text = extract_text_from_pdf(filepath)
            metadata = extract_metadata_from_filename(filename)

            if not text.strip():
                print(f"    Warning: No text extracted from {filename}")
                continue

            chunks = chunk_text(text)
            print(f"    Extracted {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                }
                documents.append(doc)

        except Exception as e:
            print(f"    Error processing {filename}: {e}")

    print(f"\nTotal documents for embedding: {len(documents)}")
    return documents
