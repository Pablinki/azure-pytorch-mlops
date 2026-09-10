# CV bullets

Every number below comes from a real run captured in `docs/evidence/` (2026-09-10).

- Designed and shipped an end-to-end MLOps pipeline on Azure: PyTorch TextCNN trained on Azure Machine
  Learning (MLflow tracking, model registry) and served via FastAPI on Azure Container Apps with scale-to-zero
  autoscaling, reaching 91.4 % accuracy / 0.914 macro-F1 on AG News (validation, 12 000 rows) with p95 latency of
  4.9 s at 100 concurrent requests on 1-vCPU replicas (p50 1.1 s, 0 errors over 2 000 requests, 1 → 4 replicas).
- Implemented a typed domain-exception hierarchy with bounded exponential-backoff retries, atomic checkpointing
  and preemption-safe resume, enforced through ruff (BLE/TRY rule sets) and 147 tests at 96.5 % coverage.
- Provisioned all infrastructure as code (Bicep) and CI/CD with GitHub Actions using OIDC federated identity,
  zero stored cloud secrets, building immutable SHA-tagged images in Azure Container Registry.

Sources: `phase5-full-job.txt` (accuracy, F1), `phase7-loadtest.txt` (latency, replicas), `phase6-local-serve.txt`
(tests, coverage), `phase8-deploy-run.txt` (OIDC pipeline).
