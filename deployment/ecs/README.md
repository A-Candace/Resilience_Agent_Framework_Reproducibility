# ECS/Fargate deployment boundary

`docker-compose.yml` is the local multi-container definition. Do not use the retired Docker Compose-to-ECS integration.
For AWS, publish the image to ECR and define separate ECS services/task definitions for:

1. `app` (public ALB target, port 8080)
2. `orchestrator` (private or authenticated API, port 8090)
3. each MCP server (private Cloud Map service discovery)
4. Phoenix, preferably as a separately managed internal service

Use Amazon RDS PostgreSQL in AWS rather than the local `postgres` container. Inject secrets from AWS Secrets Manager through ECS task definitions and assign least-privilege task roles for Bedrock, S3, and Secrets Manager.
