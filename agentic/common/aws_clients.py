import json
from functools import lru_cache
import boto3
from botocore.config import Config
from agentic.common.settings import get_settings


@lru_cache
def bedrock_client():
    cfg = get_settings()
    return boto3.client("bedrock-runtime", region_name=cfg.aws_region, config=Config(retries={"max_attempts": 3, "mode": "adaptive"}))


@lru_cache
def s3_client():
    return boto3.client("s3", region_name=get_settings().aws_region)


def get_secret(name: str) -> dict:
    cfg = get_settings()
    client = boto3.client("secretsmanager", region_name=cfg.aws_region)
    value = client.get_secret_value(SecretId=f"{cfg.secrets_prefix}/{name}")
    return json.loads(value["SecretString"])
