# NYC Resilience — Agentic MLOps Repository

This revision adds the supplied AWS agentic framework around the existing modular Streamlit application. No existing feature page was removed or rewritten. The prior monolith remains under `legacy/` for audit and rollback.

## Preserved application functions

Home, Urban Features, Risk Mapping, Socio-Demographics, Forecasting, Urban Heat Island, Multi-risk Identification Tool, Green Roof Calculator, Rain Garden Estimator, and Claude Chat remain registered in `app.py`. Automated tests verify those ten routes and their feature entry points.

## Local multi-container startup

```bash
cp .env.example .env
docker compose build
docker compose up
```

- Streamlit application: `http://localhost:8080`
- Agent orchestrator API: `http://localhost:8090/docs`
- Phoenix observability UI: `http://localhost:6006`

The application container still supports standalone execution:

```bash
docker build -t nyc-resilience .
docker run --env-file .env -p 8080:8080 nyc-resilience
```

## Architecture

- `app.py` remains the thin UI route registry.
- `src/resilience_app/features/` contains one module per existing page/button.
- `agentic/orchestrator/` provides a Bedrock-backed FastAPI/Uvicorn boundary.
- `agentic/mcp_servers/` contains bounded FastMCP servers for geospatial work, calculators, data access, and model operations.
- PostgreSQL stores durable agent-run state locally; Amazon RDS is the production target.
- Phoenix/OpenTelemetry traces agent calls; MLflow continues to track model training and GenAI experiments.
- S3 stores runtime data, training data, candidate models, and the promoted production model.
- AWS Secrets Manager and IAM task roles are the intended production credential pattern.

See:

- `docs/AGENTIC_ARCHITECTURE.md` for full flowcharts and folder mapping.
- `docs/FRAMEWORK_REVIEW.md` for design decisions and corrections.
- `docs/FUNCTIONALITY_FLOWS.md` for the original per-page flows.
- `config/agent-services.yaml` and `config/s3-layout.yaml` for service and object ownership.

## Deployment paths

The existing Beanstalk GitHub Action is retained to avoid removing the prior deployment path. The new framework is multi-container, so its preferred AWS target is ECR + ECS/Fargate. `deployment/ecs/README.md` describes the service boundary. Do not use the retired direct Docker Compose-to-ECS integration.

## Weekly training

`.github/workflows/train-weekly.yml` still performs the Monday training flow: S3 download, `training/train.py`, MLflow logging, production comparison, conditional promotion, and model reload. The existing rainfall page is rule-based; connect it to the promoted model only after an approved prediction contract is defined.

## Validation

- Existing `app.py`: byte-for-byte unchanged from the prior refactored ZIP.
- Existing feature modules: byte-for-byte unchanged from the prior refactored ZIP.
- Legacy monolith: retained.
- Python compilation: run across `app.py`, `src/`, `agentic/`, `api/`, and `training/`.
- Route and feature inventory tests: included under `tests/`.
- AWS/Bedrock/S3/RDS runtime validation requires account-specific configuration.

## Agent-owned v2 layout

The current code ownership model is documented in `docs/AGENT_OWNED_ARCHITECTURE.md`. Existing `resilience_app.features.*` imports remain supported, while new work should use `resilience_app.agents.<capability>_agent`.
