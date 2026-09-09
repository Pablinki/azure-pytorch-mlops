# CV bullets

Placeholders in square brackets are filled from real runs (`docs/evidence/`), never invented.

- Designed and shipped an end-to-end MLOps pipeline on Azure: PyTorch TextCNN trained on Azure Machine
  Learning (MLflow tracking, model registry) and served via FastAPI on Azure Container Apps with scale-to-zero
  autoscaling, reaching [ACC]% accuracy / [F1] macro-F1 on AG News with p95 latency of [X] ms at [Y] concurrent
  requests.
- Implemented a typed domain-exception hierarchy with bounded exponential-backoff retries, atomic checkpointing
  and preemption-safe resume, enforced through ruff (BLE/TRY rule sets) and [N] tests at [COV]% coverage.
- Provisioned all infrastructure as code (Bicep) and CI/CD with GitHub Actions using OIDC federated identity,
  zero stored cloud secrets, building immutable SHA-tagged images in Azure Container Registry.

Known so far (local, 2026-09-09): N = 146 tests, COV = 96.5 %. ACC / F1 / X / Y come from the cloud run.
