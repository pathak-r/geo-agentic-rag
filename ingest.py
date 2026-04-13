"""
Ingestion Script
Run once to parse PDFs and build the FAISS vector index.

Usage:
    python ingest.py
"""
from src.pdf_ingest import process_all_pdfs
from src.vector_store import build_faiss_index
from src.llm import get_embeddings


def main():
    print("=" * 60)
    print("Geo-Agentic RAG — PDF Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Process PDFs
    print("\n[1/2] Processing PDF files...")
    documents = process_all_pdfs()

    if not documents:
        print("No documents found! Place PDFs in data/pdfs/ directory.")
        return

    # Step 2: Build FAISS index
    print("\n[2/2] Building FAISS vector index...")
    embeddings = get_embeddings()
    build_faiss_index(documents, embeddings)

    print("\n" + "=" * 60)
    print("Ingestion complete! You can now run the app:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
