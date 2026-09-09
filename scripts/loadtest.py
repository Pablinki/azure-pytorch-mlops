"""Async load test for the /predict endpoint: p50/p95/p99 latency, error rate, throughput.

Usage:
    uv run python scripts/loadtest.py --url https://<fqdn> --requests 2000 --concurrency 100

While it runs, capture the replica count in another shell:
    az containerapp replica list -n textclf-api -g $AZURE_RG -o table
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Annotated

import httpx
import typer

HEADLINES = [
    "Stocks rally as tech earnings beat expectations",
    "Real Madrid wins the derby in extra time",
    "Parliament approves the new border treaty after a long debate",
    "Researchers unveil a chip that halves data-center power use",
    "Oil prices slide as investors weigh demand outlook",
    "Rocket launch delayed by weather at the coastal spaceport",
]


@dataclass
class Result:
    latencies_ms: list[float]
    errors: int
    elapsed_s: float

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return float("nan")
        data = sorted(self.latencies_ms)
        k = max(0, min(len(data) - 1, round(p / 100 * (len(data) - 1))))
        return data[k]


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    queue: asyncio.Queue[int],
    texts_per_request: int,
    latencies: list[float],
    errors: list[int],
) -> None:
    rng = random.Random()  # noqa: S311  # picking demo headlines, not security
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        payload = {"texts": rng.choices(HEADLINES, k=texts_per_request)}
        started = time.perf_counter()
        try:
            response = await client.post(f"{url}/predict", json=payload)
            ok = response.status_code == 200
        except httpx.HTTPError:  # connection reset, timeout, cold start refused, ...
            ok = False
        latencies.append((time.perf_counter() - started) * 1000)
        if not ok:
            errors.append(1)


async def run(
    url: str, requests: int, concurrency: int, texts_per_request: int, timeout: float
) -> Result:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)
    latencies: list[float] = []
    errors: list[int] = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(
            *(
                _worker(client, url, queue, texts_per_request, latencies, errors)
                for _ in range(concurrency)
            )
        )
    return Result(latencies, len(errors), time.perf_counter() - started)


def main(
    url: Annotated[
        str, typer.Option(help="Base URL, e.g. https://textclf-api.<env>.azurecontainerapps.io")
    ],
    requests: Annotated[int, typer.Option(min=1)] = 2000,
    concurrency: Annotated[int, typer.Option(min=1)] = 100,
    texts_per_request: Annotated[int, typer.Option(min=1, max=64)] = 1,
    timeout: Annotated[float, typer.Option(help="Per-request timeout in seconds")] = 30.0,
    warmup: Annotated[
        int, typer.Option(help="Sequential requests before measuring (cold start)")
    ] = 3,
) -> None:
    url = url.rstrip("/")
    with httpx.Client(timeout=timeout) as client:
        for _ in range(warmup):
            client.post(f"{url}/predict", json={"texts": [HEADLINES[0]]})
    result = asyncio.run(run(url, requests, concurrency, texts_per_request, timeout))
    summary = {
        "url": url,
        "requests": requests,
        "concurrency": concurrency,
        "texts_per_request": texts_per_request,
        "elapsed_s": round(result.elapsed_s, 2),
        "throughput_rps": round(requests / result.elapsed_s, 1),
        "p50_ms": round(result.percentile(50), 1),
        "p95_ms": round(result.percentile(95), 1),
        "p99_ms": round(result.percentile(99), 1),
        "mean_ms": round(statistics.fmean(result.latencies_ms), 1) if result.latencies_ms else None,
        "errors": result.errors,
        "error_rate": round(result.errors / requests, 4),
    }
    print(json.dumps(summary, indent=2))  # noqa: T201  # the report is the product
    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
