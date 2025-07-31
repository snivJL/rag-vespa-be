from fastapi import FastAPI, UploadFile, File, Request
from app.vespa_client import query_vespa
import shutil

app = FastAPI()


@app.post("/api/query")
async def query_rag(req: Request):
    body = await req.json()
    query = body.get("query", "")
    print(query)
    results = query_vespa(query)
    return {"chunks": results}


@app.post("/admin/upload-cert/")
async def upload_cert(file: UploadFile = File(...)):
    save_path = f"/app/certs/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}
