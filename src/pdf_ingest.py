"""
PDF Ingestion Pipeline
LlamaParse + semantic or fixed-size chunking.
"""
import os
import re
from typing import Dict, List

from llama_parse import LlamaParse
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_STRATEGY,
    LLAMA_CLOUD_API_KEY,
    MAX_CHUNK_SIZE,
    PDF_DIR,
)
from src.llm import get_embeddings


def extract_text_from_pdf(pdf_path: str, parser: LlamaParse) -> str:
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)


def extract_metadata_from_filename(filename: str) -> Dict:
    metadata = {"source_file": filename}
    name = re.sub(r"\.(pdf|PDF)(\.download)?$", "", filename, flags=re.IGNORECASE)

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

    comp_match = re.match(r"F(\d+)([A-Z])?_COMPLETION", name, re.IGNORECASE)
    if comp_match:
        well_num = comp_match.group(1)
        letter = comp_match.group(2)
        if letter:
            metadata["well_name"] = f"15/9-F-{well_num} {letter}"
        else:
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
    pdf_dir = pdf_dir or PDF_DIR
    documents: List[Dict] = []

    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return documents

    pdf_files: List[str] = []
    for f in sorted(os.listdir(pdf_dir)):
        low = f.lower()
        if not (low.endswith(".pdf") or low.endswith(".pdf.download")):
            continue
        filepath = os.path.join(pdf_dir, f)
        if not os.path.isfile(filepath):
            print(f"  Skip (not a file): {f}")
            continue
        pdf_files.append(f)

    print(f"Found {len(pdf_files)} PDF files to process (CHUNK_STRATEGY={CHUNK_STRATEGY})")

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=False,
    )
    embeddings = get_embeddings()

    semantic_chunker = None
    fixed_splitter = None
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=150,
        separators=["\n\n", "\n", "|", " ", ""],
    )

    if CHUNK_STRATEGY == "fixed":
        fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "|", " ", ""],
        )
    else:
        semantic_chunker = SemanticChunker(embeddings)

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        print(f"  Processing: {filename}")

        try:
            text = extract_text_from_pdf(filepath, parser)
            metadata = extract_metadata_from_filename(filename)

            if not text.strip():
                print(f"    Warning: No text extracted from {filename}")
                continue

            if CHUNK_STRATEGY == "fixed":
                assert fixed_splitter is not None
                chunks = fixed_splitter.split_text(text)
                print(f"    {len(chunks)} fixed-size chunks")
            else:
                assert semantic_chunker is not None
                semantic_chunks = semantic_chunker.split_text(text)
                chunks = []
                for sc in semantic_chunks:
                    if len(sc) > MAX_CHUNK_SIZE:
                        chunks.extend(secondary_splitter.split_text(sc))
                    else:
                        chunks.append(sc)
                oversized = sum(1 for sc in semantic_chunks if len(sc) > MAX_CHUNK_SIZE)
                print(
                    f"    {len(semantic_chunks)} semantic chunks → {len(chunks)} final "
                    f"({oversized} re-split >{MAX_CHUNK_SIZE} chars)"
                )

            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                })

        except Exception as e:
            print(f"    Error processing {filename}: {e}")

    print(f"\nTotal documents for embedding: {len(documents)}")
    return documents
