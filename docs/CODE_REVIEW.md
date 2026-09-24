# Careful Code Review Notes

## Preserved elements

All original page functions, field mappings, methodologies, cost sources, map rendering, download controls, Bedrock calls, MLflow GenAI logging, shapefile handling, Excel merges, and session-state routing were copied into the modular source. The untouched submitted `app.py` is retained in `legacy/app_monolith_original.py`.

## Corrections and risks identified

1. The original `requirements.txt` listed `boto3` twice with conflicting forms. It is consolidated.
2. `Procfile.txt` is renamed to the Beanstalk-recognized `Procfile`.
3. The original Docker port was 8080, despite the requested developer example using 5000.
4. Runtime data paths were relative to the current working directory. They now resolve from the repository location and can be populated from S3.
5. The existing MLflow block primarily covers GenAI traces/artifacts; it is not model-training lifecycle management.
6. GitHub scheduled workflows use UTC and do not automatically remain at 2 AM New York through daylight-saving changes. The included cron is 07:00 UTC, equal to 2 AM EST. For DST-exact 2 AM local scheduling, trigger GitHub via EventBridge Scheduler or maintain seasonal cron entries.
7. Accuracy alone may be unsuitable for imbalanced flood-risk classification. Before production, approve target definition and evaluation metrics such as recall, precision, F1, PR-AUC, calibration, and spatial/temporal holdout strategy.
8. Calling a public reload endpoint requires authentication and network controls. The scaffold uses a bearer secret; add AWS WAF/security-group restrictions or use a private authenticated trigger.
9. The application does not currently call `get_model()` for any page. Connect the approved model only after its input/output contract is defined. This prevents silently changing the existing forecasting behavior.
10. Large shapefile DBFs are duplicated (`current` and `_old`). Keep `_old` only in versioned/archive S3, not the production runtime prefix, after confirming it is unnecessary.
