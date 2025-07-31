from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from app.vespa_client import vespa_app


class VespaHybridRetriever(BaseRetriever):
    app: Vespa
    index_name: str = "passage"  # or your schema name
    pages: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        response: VespaQueryResponse = self.app.query(
            yql=f'select id, text from {self.index_name} where ([{{"targetNumHits":{self.pages}}}]nearestNeighbor(embedding, q_embedding));',
            query=query,
            hits=self.pages,
            body={
                "input.query(q_embedding)": f'embed(e5, "{query}")',
            },
            timeout="2s",
        )
        if not response.is_successful():
            raise ValueError(
                f"Query failed with status code {response.status_code}, url={response.url} response={response.json}"
            )
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        documents: List[Document] = []
        for hit in response.hits:
            fields = hit["fields"]
            documents.append(
                Document(
                    page_content=fields["text"],
                    metadata={
                        "id": fields.get("id"),
                    },
                )
            )
        return documents


vespa_retriever = VespaHybridRetriever(app=vespa_app, index_name="passage", pages=5)
