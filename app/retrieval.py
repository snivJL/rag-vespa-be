from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Dict
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from app.vespa_client import vespa_app


class VespaHybridRetriever(BaseRetriever):
    """
    Hybrid Retriever using Vespa for semantic + exact matching.
    """

    app: Vespa
    index_name: str = "doc"
    pages: int = 5
    ranking_profile: str = "hybrid"
    filters: Optional[Dict[str, str]] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Build YQL filter clause
        filter_clause = ""
        if self.filters:
            parts = [
                f"{field} contains '{value}'" for field, value in self.filters.items()
            ]
            filter_clause = " and ".join(parts) + " and "

        # Vespa YQL query
        yql = (
            f"select * from {self.index_name} where ("
            f'{filter_clause}[{{"targetNumHits":{self.pages}}}]nearestNeighbor(chunk_embedding, query_embedding)'
            ")"
        )

        response: VespaQueryResponse = self.app.query(
            yql=yql,
            query=query,
            hits=self.pages,
            ranking={self.ranking_profile},
            body={
                "input.query(query_embedding)": f'embed(e5, "{query}")',
            },
            timeout="2s",
        )
        if not response.is_successful():
            raise ValueError(
                f"Query failed with status code {response.status_code}, url={response.url}, response={response.json()}"
            )
        return self._parse_response(response)

    def _parse_response(self, response: VespaQueryResponse) -> List[Document]:
        docs: List[Document] = []
        for hit in response.hits:
            f = hit["fields"]
            # Extract core fields according to updated schema
            metadata = {
                "id": f.get("id"),
                "doc_id": f.get("doc_id"),
                "doc_type": f.get("doc_type"),
                "file_type": f.get("file_type"),
                "chunk_count": f.get("chunk_count"),
                # existing fields
                "file_name": f.get("file_name"),
                "source_url": f.get("source_url"),
                "title": f.get("title"),
                "author": f.get("author"),
                "created_timestamp": f.get("created_timestamp"),
                "modified_timestamp": f.get("modified_timestamp"),
                # pagination
                "page": f.get("page"),
                "chunk_id": f.get("chunk_id"),
                "section_title": f.get("section_title"),
                # security/access
                "visibility": f.get("visibility"),
                "allowed_users": f.get("allowed_users"),
                "allowed_groups": f.get("allowed_groups"),
                "owner": f.get("owner"),
                # excel metadata
                "sheet_name": f.get("sheet_name"),
                "row_number": f.get("row_number"),
                "column_letter": f.get("column_letter"),
                "cell_range": f.get("cell_range"),
                # business metadata
                "company": f.get("company"),
                "industry": f.get("industry"),
                "year": f.get("year"),
            }
            docs.append(
                Document(
                    page_content=f.get("chunk_text"),
                    metadata=metadata,
                )
            )
        return docs


# Usage example:
vespa_retriever = VespaHybridRetriever(app=vespa_app, index_name="doc", pages=5)
