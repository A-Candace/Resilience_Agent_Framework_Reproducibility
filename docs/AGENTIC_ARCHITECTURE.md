# Agentic Architecture and Functionality Preservation

## Design decision

The Streamlit application remains the user interface and retains all existing routes. The agentic framework is an additive service layer: the Bedrock orchestrator calls bounded FastMCP servers, while the original feature modules remain directly callable by `app.py`. This avoids turning every UI render function into a remote network dependency and protects existing behavior during the transition.

## End-to-end architecture

```mermaid
flowchart LR
    U[User] --> UI[Streamlit app :8080]
    U --> API[Agent Orchestrator :8090]
    UI --> F[Existing feature modules]
    API --> BR[Amazon Bedrock Claude]
    API --> G[MCP Geospatial :8001]
    API --> C[MCP Calculators :8002]
    API --> D[MCP Data :8003]
    API --> M[MCP Model Ops :8004]
    API --> PG[(PostgreSQL / RDS)]
    API --> PH[Phoenix / OpenTelemetry]
    D --> S3[(Amazon S3)]
    M --> S3
    BR --> API
```

## Existing functionality ownership

```mermaid
flowchart TD
    APP[app.py route registry]
    APP --> HOME[Home]
    APP --> UF[Urban Features]
    APP --> RM[Risk Mapping]
    APP --> DEM[Socio-Demographics]
    APP --> FC[Forecasting]
    APP --> UHI[Urban Heat Island]
    APP --> AI[AI Multi-Criteria Query]
    APP --> GR[Green Roof Calculator]
    APP --> RG[Rain Garden Estimator]
    APP --> CHAT[Claude Chat]

    UF -.optional tools.-> GEO[MCP Geospatial]
    RM -.optional tools.-> GEO
    DEM -.optional tools.-> GEO
    FC -.optional tools.-> GEO
    UHI -.optional tools.-> GEO
    GR -.optional tools.-> CALC[MCP Calculators]
    RG -.optional tools.-> CALC
    AI -.agent route.-> ORCH[Bedrock Orchestrator]
    CHAT -.agent route.-> ORCH
```

## Code deployment flow

```mermaid
flowchart LR
    DEV[Edit locally] --> DC[Docker Compose test]
    DC --> GH[Push to GitHub]
    GH --> CI[Compile + tests + image build]
    CI --> ECR[Push image to ECR]
    ECR --> ECS[Update ECS/Fargate services]
    ECS --> UI[App updated]
```

The prior Elastic Beanstalk workflow remains in the repository for continuity, but the uploaded framework points toward ECS/Fargate or EC2 multi-container deployment. A production migration should replace the Beanstalk step with ECR plus ECS task-definition updates.

## Weekly model training and promotion

```mermaid
flowchart TD
    CRON[GitHub Actions Monday 2 AM] --> DATA[Download training data from S3]
    DATA --> TRAIN[training/train.py]
    TRAIN --> MLF[Log metrics and artifact to MLflow]
    MLF --> CMP{Candidate better than production?}
    CMP -- No --> ALERT[Stop and alert]
    CMP -- Yes --> CAND[Write candidate artifact]
    CAND --> PROD[Promote model.pkl + metadata.json to S3 production]
    PROD --> TOOL[MCP Model Ops reload]
    TOOL --> CACHE[Atomic local model cache update]
    CACHE --> READY[New model available without UI redeployment]
```

## Repository organization

```text
app.py                              Existing thin UI router; unchanged
src/resilience_app/features/       Existing ten feature modules; unchanged
src/resilience_app/core/           Existing shared geospatial and Bedrock logic
src/resilience_app/services/       Existing S3 data/model services
agentic/
├── orchestrator/
│   ├── api.py                      FastAPI/Uvicorn agent endpoint
│   ├── bedrock.py                  Bedrock Claude client boundary
│   └── service.py                  Orchestration boundary
├── mcp_servers/
│   ├── geospatial.py               Mapping/forecast tool boundary
│   ├── calculators.py              Green roof/rain garden tool boundary
│   ├── data_access.py              Approved local/S3 data retrieval
│   └── model_ops.py                Production metadata and reload
├── common/
│   ├── settings.py                 Environment-based configuration
│   ├── aws_clients.py              Bedrock, S3, Secrets Manager
│   └── observability.py            Phoenix/OpenTelemetry setup
└── db/
    ├── models.py                   Agent run schema
    ├── session.py                  Async PostgreSQL sessions
    └── init_db.py                  Schema initialization
```

## S3 layout

```text
s3://<bucket>/nyc-resilience/
├── data/current/                   Runtime data with stable keys
├── data/versions/<date>/           Immutable snapshots
├── training/<dataset-version>/     Weekly training inputs
├── models/candidates/<run-id>/     Candidate model + metrics
├── models/production/              model.pkl + metadata.json
├── agent-artifacts/<session-id>/   Optional generated files
└── deployments/                    Release artifacts, when retained
```

## Safety and preservation controls

- `legacy/app_monolith_original.py` remains intact.
- `app.py` and every file under `src/resilience_app/features/` are byte-for-byte unchanged from the previous refactored ZIP.
- Tests assert all ten route keys and all ten feature entry-point functions.
- Phoenix initialization is fail-open so tracing outages do not stop the application.
- Agentic services are separate containers, so MCP or PostgreSQL failures do not remove the original Streamlit feature pages.
- Model reload remains a separate operation from code deployment.
