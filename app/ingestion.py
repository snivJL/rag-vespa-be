import os
import uuid
from datetime import datetime
import time

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from sentence_transformers import SentenceTransformer
from app.vespa_client import vespa_app

# Vespa endpoint and embedding model
VESPA_URL = os.getenv("VESPA_URL", "http://localhost:8080")
EMBED_MODEL = SentenceTransformer("intfloat/e5-small-v2")


def load_and_chunk(file_path: str, file_type: str) -> List:
    if file_type == "pdf":
        docs = PyPDFLoader(file_path).load()
    elif file_type == "docx":
        docs = Docx2txtLoader(file_path).load()
    elif file_type == "excel":
        docs = UnstructuredExcelLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    return splitter.split_documents(docs)


def get_embedding(text: str):
    vec = EMBED_MODEL.encode([text])[0]
    assert len(vec) == 384, f"Embedding length is {len(vec)} not 384!"
    return vec.tolist()


def prepare_chunk_for_vespa(
    chunk,
    doc_id,
    doc_type,
    file_name,
    chunk_index,
    chunk_count,
    file_type,
    extra_meta=None,
):
    # Merge chunk metadata and any extra metadata
    meta = chunk.metadata.copy()
    if extra_meta:
        meta.update(extra_meta)

    # Extract values with defaults
    page_number = meta.get("page", 0) or 0
    section_title = meta.get("section_title", "")
    sheet_name = meta.get("sheet_name", "")

    # Year fallback to current year
    try:
        year_val = int(meta.get("year", datetime.now().year))
    except Exception:
        year_val = datetime.now().year

    return {
        "id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "doc_type": doc_type,
        "file_type": file_type,
        "chunk_count": chunk_count,
        "title": meta.get("title", file_name),
        "page": page_number,
        "chunk_id": chunk_index,
        "section_title": section_title,
        "chunk_text": chunk.page_content,
        "company": meta.get("company", None),
        "industry": meta.get("industry", None),
        "year": year_val,
        # Security/Access
        "visibility": meta.get("visibility", None),
        "allowed_users": meta.get("allowed_users", []),
        "allowed_groups": meta.get("allowed_groups", []),
        "owner": meta.get("owner", None),
        # Excel Metadata
        "sheet_name": sheet_name,
        "row_number": int(meta.get("row_number", 0)),
        "column_letter": meta.get("column_letter", ""),
        "cell_range": meta.get("cell_range", ""),
    }


def ingest_doc_pyvespa(doc_fields: dict):
    response = vespa_app.feed_data_point(
        schema="doc", data_id=doc_fields["id"], fields=doc_fields
    )
    if response.is_successful():
        print(f"Successfully ingested doc {doc_fields['id']}")
    else:
        print(
            f"Failed to ingest doc {doc_fields['id']}: {response.status_code} {response.json()}"
        )


def process_and_ingest_file(
    file_path, file_type, doc_id, file_name, doc_type, extra_meta=None
):
    chunks = load_and_chunk(file_path, file_type)
    for idx, chunk in enumerate(chunks):
        doc_fields = prepare_chunk_for_vespa(
            chunk=chunk,
            doc_id=doc_id,
            doc_type=doc_type,
            file_name=file_name,
            chunk_index=idx,
            chunk_count=len(chunks),
            file_type=file_type,
            extra_meta=extra_meta,
        )
        ingest_doc_pyvespa(doc_fields)
