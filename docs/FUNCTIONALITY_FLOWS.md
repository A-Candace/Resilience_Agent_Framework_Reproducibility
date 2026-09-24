# Functionality and Pipeline Flowcharts

## 1. Integrated application routing

```mermaid
flowchart TD
    U[User opens application] --> N[ui/navigation.py sidebar]
    N --> S[Streamlit session_state.page]
    S --> A[app.py ROUTES]
    A --> H[home.py]
    A --> UF[urban_features.py]
    A --> R[risk_mapping.py]
    A --> D[demographics.py]
    A --> F[forecasting.py]
    A --> UH[urban_heat.py]
    A --> Q[ai_query.py]
    A --> C[chat.py]
    A --> G[green_roof.py]
    A --> RG[rain_garden.py]
    UF & R & D & UH & Q & RG --> DL[services/data_loader.py]
    DL --> DS[data/static or S3-synced files]
    UF & R & D & F & UH & Q & C & G & RG --> SH[core/shared.py]
```

## 2. Urban Features

```mermaid
flowchart LR
    B[Urban Features button] --> P[features/urban_features.py]
    P --> L[ensure_boards]
    L --> SHP[NYC tract shapefile]
    L --> XLSX[Risk Attributes Excel]
    P --> M[PyDeck tract map]
    P --> A[Choose elevation, slope, or building attribute]
    A --> C[Choropleth + legend]
    C --> O[PNG and CSV downloads]
```

## 3. Risk Mapping

```mermaid
flowchart LR
    B[Risk Mapping button] --> P[features/risk_mapping.py]
    P --> L[Load merged tract attributes]
    L --> T{Risk choice}
    T --> NRI[FEMA NRI fields]
    T --> CMP[Composite/rescaled components]
    NRI & CMP --> Q[Quantile bins and colors]
    Q --> M[PyDeck choropleth]
    M --> O[PNG and CSV downloads]
```

## 4. Socio-Demographics

```mermaid
flowchart LR
    B[Socio-Demographics button] --> P[features/demographics.py]
    P --> C[Select demographic category]
    C --> F[Select mapped Excel field]
    F --> L[Load merged tract data]
    L --> Q[Quantile classification]
    Q --> M[Map + methodology]
    M --> O[PNG and CSV downloads]
```

## 5. Rainfall Forecasting — current rule engine

```mermaid
flowchart LR
    B[Forecasting button] --> P[features/forecasting.py]
    P --> I[User enters rainfall inches]
    I --> G[Load 4 km historical flood grid GeoJSON]
    G --> R[Trigger cells where forecast >= historical minimum flood rainfall]
    R --> H[Heatmap weighted by historical event count]
    H --> T[Triggered-cell table]
```

This is preserved behavior and is not currently connected to `model.pkl`.

## 6. Urban Heat Island

```mermaid
flowchart LR
    B[Urban Heat button] --> P[features/urban_heat.py]
    P --> L[Load census tract + NRI attributes]
    L --> H[Select HWAV heat-risk field]
    H --> Q[Quantile bins]
    Q --> M[Heat choropleth]
    M --> O[PNG and CSV downloads]
```

## 7. AI Multi-Criteria Query

```mermaid
flowchart TD
    B[Multi-risk Identification Tool button] --> P[features/ai_query.py]
    P --> Q[User natural-language criteria]
    Q --> BR[Bedrock Claude parser]
    BR --> J[Structured JSON criteria]
    J --> E[core query filter engine]
    E --> T[Matching census tracts]
    T --> M[Highlighted map]
    T --> D[Results table and CSV]
    BR --> ML[MLflow GenAI trace when configured]
```

## 8. Chat

```mermaid
flowchart LR
    B[Chat button] --> P[features/chat.py]
    P --> H[Session chat history]
    H --> BR[Claude through Bedrock]
    BR --> R[Response rendered in Streamlit]
    BR --> ML[MLflow GenAI trace/artifacts]
```

## 9. Green Roof Calculator

```mermaid
flowchart LR
    B[Green Roof button] --> P[features/green_roof.py]
    P --> A[Area input]
    A --> U[Default or overridden unit-cost range]
    U --> C[Low and high cost calculation]
    C --> S[Source reference]
```

## 10. Rain Garden Estimator

```mermaid
flowchart TD
    B[Rain Garden button] --> P[features/rain_garden.py]
    P --> A[Area and unit-cost estimate]
    P --> L[Load tract elevation and slope]
    L --> R{Candidate rule}
    R --> S[Low slope]
    R --> E[Low elevation]
    R --> SE[Both]
    S & E & SE --> PC[Percentile cutoff]
    PC --> M[Green candidate tract map]
```

## 11. Code deployment workflow

```mermaid
flowchart LR
    D[Developer edits modular files] --> T[Local Docker test on port 8080]
    T --> G[Push to GitHub main]
    G --> A[deploy.yml]
    A --> C[Compile and Docker build]
    C --> Z[Create Beanstalk ZIP]
    Z --> S3[Upload deployment artifact to S3]
    S3 --> V[Create Beanstalk application version]
    V --> EB[Update Beanstalk environment]
```

## 12. Weekly model workflow

```mermaid
flowchart TD
    CRON[Monday scheduled GitHub Action] --> DATA[Download training data from S3]
    DATA --> TRAIN[training/train.py]
    TRAIN --> MF[Log params, metrics, candidate artifact to MLflow]
    MF --> CMP{Candidate accuracy > production metadata accuracy?}
    CMP -- No --> ALERT[Stop promotion and send/attach alert integration]
    CMP -- Yes --> UP[Upload model.pkl + metadata.json to production S3 prefix]
    UP --> API[POST Beanstalk /reload-model]
    API --> DL[FastAPI downloads production model from S3]
    DL --> SWAP[Thread-safe in-memory model swap]
    SWAP --> DONE[No application redeployment]
```
