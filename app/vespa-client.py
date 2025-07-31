from vespa.application import Vespa
import os

vespa_app = Vespa(url=os.getenv("VESPA_HOST"))


def query_vespa(query: str):
    response = vespa_app.query(
        {"yql": f'select * from sources * where content contains "{query}";', "hits": 5}
    )
    return [hit["fields"]["content"] for hit in response.hits]
