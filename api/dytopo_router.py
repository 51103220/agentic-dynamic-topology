"""
DyTopo FastAPI router — mounted at /dytopo in main.py
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel

from dytopo_engine import DyTopoConfig, RoundResult, run_dytopo

router = APIRouter()

# ── Prometheus metrics ────────────────────────────────────────────────────────
dytopo_jobs_total       = Counter("dytopo_jobs_total", "Total DyTopo jobs submitted", ["task_type", "mode"])
dytopo_jobs_active      = Gauge("dytopo_jobs_active", "Currently running DyTopo jobs")
dytopo_rounds_total     = Counter("dytopo_rounds_total", "Total reasoning rounds executed", ["task_type"])
dytopo_job_duration     = Histogram("dytopo_job_duration_seconds", "Job wall-clock time",
                                    buckets=[1, 2, 5, 10, 20, 30, 60])
dytopo_edge_density     = Histogram("dytopo_round_edge_density", "Graph edge density per round (0-1)",
                                    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
dytopo_agent_pool_size  = Gauge("dytopo_agent_pool_size", "Number of agents in latest job")

# ── In-memory job store ───────────────────────────────────────────────────────
class JobState(BaseModel):
    job_id: str
    status: str = "pending"          # pending | running | done | error
    rounds: list[dict] = []
    final_answer: str | None = None
    error: str | None = None

_jobs: dict[str, JobState] = {}
_job_queues: dict[str, asyncio.Queue] = {}

# ── Request / response models ─────────────────────────────────────────────────
class RunRequest(BaseModel):
    problem: str
    task_type: str = "code"          # "code" | "math"
    n_agents: int = 4
    tau: float = 0.3
    max_rounds: int = 5
    mock: bool = True
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"


class RunResponse(BaseModel):
    job_id: str


# ── Background worker ─────────────────────────────────────────────────────────
async def _execute_job(job_id: str, problem: str, cfg: DyTopoConfig):
    job = _jobs[job_id]
    queue = _job_queues[job_id]
    job.status = "running"
    dytopo_jobs_active.inc()
    dytopo_agent_pool_size.set(cfg.n_agents)
    t0 = time.monotonic()

    try:
        async for round_result in run_dytopo(problem, cfg):
            d = round_result.to_dict()
            job.rounds.append(d)
            if round_result.final_answer:
                job.final_answer = round_result.final_answer

            # record metrics per round
            dytopo_rounds_total.labels(task_type=cfg.task_type).inc()
            N = cfg.n_agents
            total_possible = N * (N - 1)
            if total_possible > 0:
                active_edges = sum(
                    round_result.adjacency[j][i]
                    for j in range(N) for i in range(N) if j != i
                )
                dytopo_edge_density.observe(active_edges / total_possible)

            await queue.put(("round", d))

        job.status = "done"
        await queue.put(("done", None))

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await queue.put(("error", str(exc)))

    finally:
        dytopo_jobs_active.dec()
        dytopo_job_duration.observe(time.monotonic() - t0)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/run", response_model=RunResponse)
async def run_dytopo_job(req: RunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    cfg = DyTopoConfig(
        task_type=req.task_type,
        n_agents=req.n_agents,
        tau=req.tau,
        max_rounds=req.max_rounds,
        mock_mode=req.mock,
        llm_base_url=req.llm_base_url,
        llm_model=req.llm_model,
    )
    _jobs[job_id] = JobState(job_id=job_id)
    _job_queues[job_id] = asyncio.Queue()
    dytopo_jobs_total.labels(task_type=req.task_type, mode="mock" if req.mock else "llm").inc()
    background_tasks.add_task(_execute_job, job_id, req.problem, cfg)
    return RunResponse(job_id=job_id)


@router.get("/job/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE endpoint — streams RoundResult events as each round completes."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = _job_queues[job_id]

    async def event_generator():
        while True:
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"
                continue

            if kind == "round":
                yield f"event: round\ndata: {json.dumps(data)}\n\n"
            elif kind == "done":
                yield "event: done\ndata: {}\n\n"
                break
            elif kind == "error":
                yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/job/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    """Polling fallback — returns full job state."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id].model_dump()


@router.get("/presets")
async def get_presets():
    return [
        {
            "id": "two-sum",
            "label": "Two Sum (Code)",
            "task_type": "code",
            "n_agents": 4,
            "tau": 0.3,
            "problem": "Write a Python function that takes a list of integers `nums` and a target integer `target`, and returns the indices of the two numbers that add up to the target. Each input has exactly one solution. Example: nums=[2,7,11,15], target=9 → [0,1].",
        },
        {
            "id": "fizzbuzz",
            "label": "FizzBuzz (Code)",
            "task_type": "code",
            "n_agents": 4,
            "tau": 0.3,
            "problem": "Write a Python function fizzbuzz(n) that returns a list of strings for numbers 1 to n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, and the number as string otherwise.",
        },
        {
            "id": "diophantine",
            "label": "Diophantine Equation (Math)",
            "task_type": "math",
            "n_agents": 4,
            "tau": 0.4,
            "problem": "Find all positive integer solutions to the equation x² + y² = z² where x < y < z and z ≤ 20. List all Pythagorean triples and verify each one.",
        },
        {
            "id": "series-sum",
            "label": "Series Sum (Math)",
            "task_type": "math",
            "n_agents": 4,
            "tau": 0.4,
            "problem": "Calculate the sum of the infinite geometric series: 1 + 1/2 + 1/4 + 1/8 + ... Prove convergence, derive the closed-form formula, and compute the exact sum.",
        },
    ]


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Aggregate statistics for the React metrics dashboard."""
    jobs = list(_jobs.values())
    total = len(jobs)
    active = sum(1 for j in jobs if j.status == "running")
    done   = sum(1 for j in jobs if j.status == "done")
    error  = sum(1 for j in jobs if j.status == "error")

    all_rounds = [r for j in jobs for r in j.rounds]
    total_rounds = len(all_rounds)

    # edge density samples per round (list of floats 0-1)
    density_samples: list[float] = []
    for r in all_rounds:
        adj = r.get("adjacency", [])
        N = len(adj)
        if N < 2:
            continue
        total_possible = N * (N - 1)
        active_edges = sum(adj[j][i] for j in range(N) for i in range(N) if j != i)
        density_samples.append(active_edges / total_possible)

    # per-round density indexed by round number (0,1,2) — avg across all jobs
    round_densities: dict[int, list[float]] = {}
    for r in all_rounds:
        adj = r.get("adjacency", [])
        N = len(adj)
        rnum = r.get("round", 0)
        if N < 2:
            continue
        total_possible = N * (N - 1)
        active_edges = sum(adj[j][i] for j in range(N) for i in range(N) if j != i)
        round_densities.setdefault(rnum, []).append(active_edges / total_possible)
    avg_density_by_round = {
        k: sum(v) / len(v) for k, v in sorted(round_densities.items())
    }

    # task type breakdown
    task_counts = {"code": 0, "math": 0}
    mode_counts = {"mock": 0, "llm": 0}
    for j in jobs:
        # infer task_type from first round agents
        if j.rounds:
            roles = [a.get("role", "") for a in j.rounds[0].get("agents", [])]
            if any(r in ("Developer", "Researcher", "Tester", "Designer") for r in roles):
                task_counts["code"] += 1
            else:
                task_counts["math"] += 1

    # recent jobs (last 20), newest first
    recent = [
        {
            "job_id": j.job_id,
            "status": j.status,
            "rounds": len(j.rounds),
            "final_answer_len": len(j.final_answer or ""),
        }
        for j in jobs[-20:][::-1]
    ]

    return {
        "total_jobs": total,
        "active_jobs": active,
        "done_jobs": done,
        "error_jobs": error,
        "total_rounds": total_rounds,
        "avg_rounds_per_job": total_rounds / done if done > 0 else 0,
        "avg_edge_density": sum(density_samples) / len(density_samples) if density_samples else 0,
        "avg_density_by_round": avg_density_by_round,
        "task_counts": task_counts,
        "density_samples": density_samples[-50:],  # last 50 for histogram
        "recent_jobs": recent,
    }
