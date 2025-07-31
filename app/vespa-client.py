from vespa.application import Vespa

vespa_app = Vespa(url="http://your-vespa-host:8080")  # use Railway ENV later


def query_vespa(query: str):
    response = vespa_app.query(
        {"yql": f'select * from sources * where content contains "{query}";', "hits": 5}
    )
    return [hit["fields"]["content"] for hit in response.hits]
