"""
PDF Ingestion Pipeline
Parses well reports and drilling reports using LlamaParse + SemanticChunker.
"""
import os
import re
from typing import List, Dict
from llama_parse import LlamaParse
from langchain_experimental.text_splitter import SemanticChunker
from src.config import PDF_DIR, LLAMA_CLOUD_API_KEY
from src.llm import get_embeddings


def extract_text_from_pdf(pdf_path: str, parser: LlamaParse) -> str:
    """Extract structured markdown text from a PDF file using LlamaParse."""
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)


def extract_metadata_from_filename(filename: str) -> Dict:
    """
    Extract well name and date from PDF filename.
    Examples:
        15_9_F_11_2013_03_08.pdf -> well=15/9-F-11, date=2013-03-08
        15_9_19_A_1997_07_30.pdf -> well=15/9-19A, date=1997-07-30
        F12_COMPLETION_REPORT_1.PDF -> well=F-12, type=completion_report
    """
    metadata = {"source_file": filename}

    name = filename.replace(".pdf", "").replace(".PDF", "")

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

    comp_match = re.match(r"F(\d+)_COMPLETION", name, re.IGNORECASE)
    if comp_match:
        well_num = comp_match.group(1)
        metadata["well_name"] = f"15/9-F-{well_num}"
        metadata["doc_type"] = "completion_report"
        return metadata

    fwr_match = re.match(r"FWR_Completion_F(\d+)", name, re.IGNORECASE)
    if fwr_match:
        well_num = fwr_match.group(1)
        metadata["well_name"] = f"15/9-F-{well_num}"
        metadata["doc_type"] = "final_well_report"
        return metadata

    fc_match = re.match(r"15-9-F-(\d+)-([A-Z]+)", name, re.IGNORECASE)
    if fc_match:
        well_num = fc_match.group(1)
        sidetrack = fc_match.group(2)
        metadata["well_name"] = f"15/9-F-{well_num} {sidetrack}"
        metadata["doc_type"] = "completion_report"
        return metadata

    metadata["well_name"] = "Unknown"
    metadata["doc_type"] = "unknown"
    return metadata


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

    pdf_files = [f for f in os.listdir(pdf_dir)
                 if f.lower().endswith(".pdf")]

    print(f"Found {len(pdf_files)} PDF files to process")

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=False,
    )
    embeddings = get_embeddings()
    chunker = SemanticChunker(embeddings)

    for filename in sorted(pdf_files):
        filepath = os.path.join(pdf_dir, filename)
        print(f"  Processing: {filename}")

        try:
            text = extract_text_from_pdf(filepath, parser)
            metadata = extract_metadata_from_filename(filename)

            if not text.strip():
                print(f"    Warning: No text extracted from {filename}")
                continue

            chunks = chunker.split_text(text)
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
