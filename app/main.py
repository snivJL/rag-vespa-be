from fastapi import FastAPI, UploadFile, File
from app.retrieval import vespa_retriever
from pydantic import BaseModel
import shutil

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


@app.post("/admin/upload-cert/")
async def upload_cert(file: UploadFile = File(...)):
    save_path = f"/app/certs/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}
