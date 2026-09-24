from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NYC Resilience Agentic Platform"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    database_url: str = "postgresql+asyncpg://resilience:resilience@postgres:5432/resilience"
    s3_bucket: str = ""
    secrets_prefix: str = "nyc-resilience"
    phoenix_collector_endpoint: str = "http://phoenix:4317"
    otel_service_name: str = "nyc-resilience-orchestrator"
    mcp_geospatial_url: str = "http://mcp-geospatial:8001/mcp"
    mcp_calculators_url: str = "http://mcp-calculators:8002/mcp"
    mcp_data_url: str = "http://mcp-data:8003/mcp"
    mcp_model_url: str = "http://mcp-model:8004/mcp"
    mcp_forecasting_url: str = "http://mcp-forecasting:8012/mcp"
    orchestrator_url: str = "http://orchestrator:8090"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
