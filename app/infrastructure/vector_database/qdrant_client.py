from qdrant_client import QdrantClient


class QdrantClientWrapper:
    """
    Quản lý kết nối Qdrant.
    Không chứa logic retrieval nghiệp vụ.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ):
        self.url = url
        self.api_key = api_key or None

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.client.close()