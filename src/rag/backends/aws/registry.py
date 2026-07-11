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

    def load_all(self) -> dict:
        items = self._table.scan()["Items"]
        # Rebuild the registry key from stored attributes (matches BookEntry.registry_key);
        # registry_key itself is a computed property and not persisted on the item.
        return {"books": {f"{i['content_hash']}_{i['pipeline_config_hash']}": i for i in items}}
