import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

import boto3
from botocore.client import Config
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dytopo_router import router as dytopo_router
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
import redis.asyncio as aioredis

REDIS_HOST    = os.getenv("REDIS_HOST",    "redis")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
MINIO_ENDPOINT= os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_KEY     = os.getenv("MINIO_ACCESS_KEY",  "minioadmin")
MINIO_SECRET  = os.getenv("MINIO_SECRET_KEY",  "minioadmin")
MINIO_BUCKET  = os.getenv("MINIO_BUCKET",      "bus-history")

redis_pool: aioredis.Redis | None = None
ws_clients: set[WebSocket] = set()

fleet_active_buses = Gauge("bus_fleet_active_buses", "Number of active buses in Redis")
fleet_active_routes = Gauge("bus_fleet_active_routes", "Number of active routes represented by active buses")
route_avg_speed = Gauge("bus_route_avg_speed_kmh", "Rolling average speed by route", ["route_no"])
route_vehicle_count = Gauge("bus_route_vehicle_count", "Rolling vehicle count by route speed window", ["route_no"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool
    redis_pool = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    asyncio.create_task(pubsub_relay())
    yield
    await redis_pool.aclose()


app = FastAPI(title="Bus Tracking API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3002",
        "http://localhost:5173",
        "http://localhost:4173",
        "http://dashboard",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dytopo_router, prefix="/dytopo", tags=["dytopo"])


async def pubsub_relay():
    """Subscribe to Redis bus-updates channel and fan-out to all WS clients."""
    global ws_clients
    pubsub = redis_pool.pubsub()
    await pubsub.subscribe("bus-updates")
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        if not ws_clients:
            continue
        dead = set()
        for ws in list(ws_clients):
            try:
                await ws.send_text(message["data"])
            except Exception:
                dead.add(ws)
        ws_clients -= dead


@app.get("/buses")
async def get_all_buses() -> list[dict[str, Any]]:
    result = []
    cursor = 0
    while True:
        cursor, keys = await redis_pool.scan(cursor, match="bus:*", count=200)
        for key in keys:
            data = await redis_pool.hgetall(key)
            if data:
                result.append(data)
        if cursor == 0:
            break
    return result


@app.get("/bus/{vehicle_id}")
async def get_bus(vehicle_id: str) -> dict[str, Any]:
    data = await redis_pool.hgetall(f"bus:{vehicle_id}")
    return data or {}


@app.get("/bus/{vehicle_id}/history")
async def get_bus_history(vehicle_id: str, date: str | None = None) -> list[str]:
    s3 = _s3_client()
    prefix = date if date else ""
    try:
        resp = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix, MaxKeys=200)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as e:
        return [f"error: {e}"]


@app.get("/routes")
async def get_routes() -> list[dict[str, str]]:
    # lấy danh sách tuyến từ sorted set active-buses
    vehicles = await redis_pool.zrange("active-buses", 0, -1)
    routes: dict[str, str] = {}
    for v in vehicles:
        data = await redis_pool.hgetall(f"bus:{v}")
        rno = data.get("route_no")
        rid = data.get("route_id")
        if rno and rno not in routes:
            routes[rno] = rid or ""
    return [{"route_no": k, "route_id": v} for k, v in sorted(routes.items())]


@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    active = await redis_pool.zcard("active-buses")
    info   = await redis_pool.info("stats")
    return {
        "active_buses": active,
        "total_commands_processed": info.get("total_commands_processed", 0),
        "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
    }

@app.get("/metrics")
async def metrics() -> Response:
    """Expose business metrics for Prometheus/Grafana."""
    active = await redis_pool.zcard("active-buses")
    fleet_active_buses.set(active)

    routes = set()
    vehicles = await redis_pool.zrange("active-buses", 0, -1)
    for vehicle in vehicles:
        data = await redis_pool.hgetall(f"bus:{vehicle}")
        route = data.get("route_no")
        if route:
            routes.add(route)
    fleet_active_routes.set(len(routes))

    cursor = 0
    while True:
        cursor, keys = await redis_pool.scan(cursor, match="route:speed:*", count=200)
        for key in keys:
            data = await redis_pool.hgetall(key)
            route = key.removeprefix("route:speed:")
            if data.get("avgSpeed") is not None:
                route_avg_speed.labels(route).set(float(data["avgSpeed"]))
            if data.get("vehicleCount") is not None:
                route_vehicle_count.labels(route).set(float(data["vehicleCount"]))
        if cursor == 0:
            break

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── WebSocket endpoint ─────────────────────────────────────────────────────

@app.websocket("/ws/buses")
async def ws_buses(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            # giữ kết nối sống, data đến từ pubsub
            await asyncio.sleep(30)
            await ws.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_KEY,
        aws_secret_access_key=MINIO_SECRET,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
