import boto3
from boto3.dynamodb.conditions import Attr

from rag.core.schemas import BookEntry
from rag.config import AWS_REGION, DYNAMODB_TABLE


class DynamoDBRegistry:
    """Book ingestion ledger backed by DynamoDB.

    Table schema:
      PK  content_hash        (String) — SHA-256 of the raw source file
      SK  pipeline_config_hash (String) — fingerprint of all pipeline method decisions

    Conditional writes prevent duplicate registration under concurrent ingestion.
    """

    def __init__(self) -> None:
        self._table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)

    def is_ingested(self, content_hash: str, pipeline_config_hash: str) -> bool:
        r = self._table.get_item(
            Key={"content_hash": content_hash, "pipeline_config_hash": pipeline_config_hash},
            ProjectionExpression="content_hash",
        )
        return "Item" in r

    def register(self, entry: BookEntry) -> None:
        self._table.put_item(
            Item=entry.to_dict(),
            ConditionExpression=Attr("content_hash").not_exists(),
        )

    def get(self, content_hash: str, pipeline_config_hash: str) -> dict:
        r = self._table.get_item(
            Key={"content_hash": content_hash, "pipeline_config_hash": pipeline_config_hash}
        )
        return r.get("Item", {})

    def find_superseded(self, filename: str, pipeline_config_hash: str, content_hash: str) -> list[str]:
        """Older versions of the same document: same filename + config, different content.

        Cold-path admin scan (paginated). Fine at current scale; a GSI on `filename`
        is the path if the table grows large.
        """
        hashes: list[str] = []
        kwargs = {
            "FilterExpression": Attr("filename").eq(filename)
            & Attr("pipeline_config_hash").eq(pipeline_config_hash)
            & Attr("content_hash").ne(content_hash),
            "ProjectionExpression": "content_hash",
        }
        while True:
            resp = self._table.scan(**kwargs)
            hashes.extend(i["content_hash"] for i in resp["Items"])
            if "LastEvaluatedKey" not in resp:
                return hashes
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def delete(self, content_hash: str, pipeline_config_hash: str) -> None:
        self._table.delete_item(
            Key={"content_hash": content_hash, "pipeline_config_hash": pipeline_config_hash}
        )
