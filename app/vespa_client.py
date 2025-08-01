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
