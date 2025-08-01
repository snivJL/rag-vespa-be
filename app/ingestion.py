import os

import uuid
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from sentence_transformers import SentenceTransformer
from app.vespa_client import vespa_app


# Set up your environment variable: VESPA_URL
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
    chunks = splitter.split_documents(docs)
    return chunks


def get_embedding(text: str):
    vec = EMBED_MODEL.encode([text])[0]
    assert len(vec) == 384, f"Embedding length is {len(vec)} not 384!"
    return vec.tolist()


def prepare_chunk_for_vespa(chunk, doc_id, doc_type, file_name, extra_meta=None):
    meta = chunk.metadata.copy()
    if extra_meta:
        meta.update(extra_meta)
    page_number = meta.get("page", 0) or 0
    sheet_name = meta.get("sheet_name", "")

    return {
        "id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "doc_type": doc_type,
        "file_name": file_name,
        "source_url": meta.get("source", file_name),
        "title": meta.get("title", file_name),
        "author": meta.get("author", ""),
        "created_at": meta.get("created_at", datetime.utcnow().isoformat()),
        "modified_at": meta.get("modified_at", datetime.utcnow().isoformat()),
        "page_number": page_number,
        "sheet_name": sheet_name,
        "text": chunk.page_content,
    }


def ingest_passage_pyvespa(doc_fields: dict, doc_id: str):
    response = vespa_app.feed_data_point(
        schema="passage", data_id=doc_id, fields=doc_fields
    )
    if response.is_successful():
        print(f"Successfully ingested doc {doc_id}")
    else:
        print(f"Failed to ingest doc {doc_id}: {response.status_code} {response.json}")


def process_and_ingest_file(
    file_path, file_type, doc_id, file_name, doc_type, extra_meta=None
):
    chunks = load_and_chunk(file_path, file_type)
    for chunk in chunks:
        doc_fields = prepare_chunk_for_vespa(
            chunk=chunk,
            doc_id=doc_id,
            doc_type=doc_type,
            file_name=file_name,
            extra_meta=extra_meta,
        )
        # Compute embedding for chunk
        # doc_fields["embedding"] = get_embedding(doc_fields["text"])
        ingest_passage_pyvespa(doc_fields, doc_fields["id"])
