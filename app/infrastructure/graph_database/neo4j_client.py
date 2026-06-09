from neo4j import GraphDatabase, Driver, Session


class Neo4jClient:
    """
    Quản lý kết nối Neo4j.
    Không chứa logic nghiệp vụ.
    Không chứa Cypher dài.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        self.driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

    def session(self) -> Session:
        return self.driver.session(database=self.database)

    def close(self) -> None:
        self.driver.close()

    def health_check(self) -> bool:
        try:
            with self.session() as session:
                result = session.run("RETURN 1 AS ok")
                record = result.single()
                return record is not None and record["ok"] == 1
        except Exception:
            return False