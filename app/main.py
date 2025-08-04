from typing import List, Optional
from app.ingestion import process_and_ingest_file
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.retrieval import vespa_retriever
from pydantic import BaseModel
import tempfile
import shutil
import uuid
import logging
from pprint import pformat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/query")
def query_docs(payload: QueryRequest):
    logger.info("User Query sent: %s", payload.query)
    docs = vespa_retriever.get_relevant_documents(payload.query)
    formatted_docs = [
        {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]

    logger.info("Retrieved documents:\n%s", pformat(formatted_docs))

    return {"documents": formatted_docs}


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


@app.post("/admin/upload-cert/")
async def upload_cert(file: UploadFile = File(...)):
    save_path = f"/app/certs/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}
