import os
from vespa.application import Vespa

VESPA_URL = os.getenv("VESPA_URL")
cert_path = os.getenv("VESPA_CERT", "/app/certs/data-plane-public-cert.pem")
key_path = os.getenv("VESPA_KEY", "/app/certs/data-plane-private-key.pem")
LLM_API_KEY = os.getenv("VESPA_LLM_API_KEY")

vespa_app = Vespa(
    url=VESPA_URL,
    cert=cert_path,
    key=key_path,
)


# def query_vespa(query_text: str):
#     response = vespa_app.query(
#         body={
#             "queryProfile": "vector-search",
#             "query_text": query_text,
#             "hits": 5,
#             "input.query(q_embedding)": f"embed(@query_text)",
#             "prompt": "@query -  Answer to the query in the details, list all the documents provided and its content. See documents below:{context}",
#             "traceLevel": 1,
#         },
#         headers={"X-LLM-API-KEY": LLM_API_KEY},
#     )
#     return response.hits
