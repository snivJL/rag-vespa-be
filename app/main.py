from fastapi import FastAPI, Request
from app.vespa_client import query_vespa

app = FastAPI()


@app.post("/api/query")
async def query_rag(req: Request):
    body = await req.json()
    query = body.get("query", "")
    print(query)
    results = query_vespa(query)
    return {"chunks": results}
