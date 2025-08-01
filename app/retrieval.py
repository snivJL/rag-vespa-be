from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Dict
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from app.vespa_client import vespa_app


class VespaHybridRetriever(BaseRetriever):
    app: Vespa
    index_name: str = "passage"
    pages: int = 5
    ranking_profile: str = "hybrid"
    filters: Optional[Dict[str, str]] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        filter_clause = ""
        if self.filters:
            filter_parts = [
                f"{field} contains '{value}'" for field, value in self.filters.items()
            ]
            filter_clause = " and ".join(filter_parts) + " and "
        yql = (
            f"select * from {self.index_name} where ("
            f'{filter_clause}([{{"targetNumHits":{self.pages}}}]nearestNeighbor(embedding, q_embedding))'
            ");"
        )
        response: VespaQueryResponse = self.app.query(
            yql=yql,
            query=query,
            hits=self.pages,
            ranking={self.ranking_profile},
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
            metadata = {
                "id": fields.get("id"),
                "doc_id": fields.get("doc_id"),
                "doc_type": fields.get("doc_type"),
                "file_name": fields.get("file_name"),
                "source_url": fields.get("source_url"),
                "title": fields.get("title"),
                "author": fields.get("author"),
                "created_at": fields.get("created_at"),
                "modified_at": fields.get("modified_at"),
                "page_number": fields.get("page_number"),
                "sheet_name": fields.get("sheet_name"),
                # Add more fields if needed
            }
            documents.append(
                Document(
                    page_content=fields["text"],
                    metadata=metadata,
                )
            )
        return documents


# Usage:
vespa_retriever = VespaHybridRetriever(app=vespa_app, index_name="passage", pages=5)
