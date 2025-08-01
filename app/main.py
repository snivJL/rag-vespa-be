from typing import List, Optional
from app.ingestion import process_and_ingest_file
from fastapi import FastAPI, UploadFile, File, Form
from app.retrieval import vespa_retriever
from pydantic import BaseModel
import tempfile
import shutil
import uuid

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/query")
def query_docs(payload: QueryRequest):
    docs = vespa_retriever.get_relevant_documents(payload.query)
    return {
        "documents": [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]
    }


@app.post("/api/ingest_file")
async def ingest_file(
    file: UploadFile = File(...),
    file_type: str = Form(...),  # "pdf", "docx", "excel"
    doc_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
):
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Compose file/document metadata
    file_name = file.filename
    doc_type = file_type
    if not doc_id:
        doc_id = str(uuid.uuid4())

    extra_meta = {
        "title": title or file_name,
        "author": author or "",
        "source_url": source_url or file_name,
    }

    # Ingest file (this handles chunking, embedding, Vespa insert)
    process_and_ingest_file(
        file_path=tmp_path,
        file_type=file_type,
        doc_id=doc_id,
        file_name=file_name,
        doc_type=doc_type,
        extra_meta=extra_meta,
    )

    return {"status": "success", "doc_id": doc_id}
