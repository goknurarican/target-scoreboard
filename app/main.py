# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
VantAI Target Scoreboard FastAPI Application - Phase 1C Production.
Features: DI wiring, structured logging, healthcheck, circuit breaker scaffold.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    ScoreRequest,
    ScoreResponse,
    HealthCheckResponse,
    ErrorResponse
)
from .scoring import score_targets, validate_score_request
from .data_access.opentargets import get_ot_client, cleanup_ot_client
from .data_access.cache import get_cache_manager, cleanup_cache, cache_stats
from .data_access.stringdb import get_stringdb_client, cleanup_stringdb_client

logger = logging.getLogger(__name__)

# ========================
# Configuration from Environment
# ========================

# Service configuration
SERVICE_NAME = os.getenv("SERVICE_NAME", "VantAI Target Scoreboard")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0-phase1c")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# CORS configuration - Updated with common dev ports
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000,http://localhost:5173,http://localhost:8000").split(",")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

# Circuit breaker configuration
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "false").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json or text

# Default weights configuration
DEFAULT_WEIGHTS = {
    "genetics": float(os.getenv("WEIGHT_GENETICS", "0.35")),
    "ppi": float(os.getenv("WEIGHT_PPI", "0.25")),
    "pathway": float(os.getenv("WEIGHT_PATHWAY", "0.20")),
    "safety": float(os.getenv("WEIGHT_SAFETY", "0.10")),
    "modality_fit": float(os.getenv("WEIGHT_MODALITY", "0.10")),
}


# ========================
# Structured Logging Setup
# ========================

def setup_logging():
    """Configure structured logging with request IDs."""
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    if LOG_FORMAT == "json":
        # JSON formatter for production
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "service": SERVICE_NAME,
                    "version": SERVICE_VERSION
                }

                # Add extra fields if present
                if hasattr(record, "extra"):
                    log_record.update(record.extra)

                # Add exception info if present
                if record.exc_info:
                    log_record["exception"] = self.formatException(record.exc_info)

                return json.dumps(log_record)

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
    else:
        # Text formatter for development
        format_str = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_str))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)  # DEBUG aşamasında INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ========================
# Dependency Injection Container
# ========================

class DIContainer:
    """Simple dependency injection container for clients and services."""

    def __init__(self):
        self._clients = {}
        self._services = {}
        self._initialized = False

    async def initialize(self):
        """Initialize all clients and services at startup."""
        if self._initialized:
            return

        logger = logging.getLogger(__name__)
        start_time = time.time()

        try:
            # Initialize OpenTargets client
            self._clients["opentargets"] = await get_ot_client()
            logger.info("OpenTargets client initialized")

            # Initialize cache manager
            self._services["cache"] = await get_cache_manager()
            logger.info("Cache manager initialized")

            # Initialize StringDB client
            self._clients["stringdb"] = await get_stringdb_client()
            logger.info("STRING-DB client initialized")

            self._initialized = True
            init_time = (time.time() - start_time) * 1000

            logger.info(
                f"DI container initialized in {init_time:.1f}ms",
                extra={
                    "clients_count": len(self._clients),
                    "services_count": len(self._services),
                    "init_time_ms": init_time
                }
            )

        except Exception as e:
            logger.error(f"DI container initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup all clients and services at shutdown."""
        logger = logging.getLogger(__name__)

        # Cleanup clients
        if "opentargets" in self._clients:
            await cleanup_ot_client()

        if "stringdb" in self._clients:
            await cleanup_stringdb_client()

        # Cleanup services
        if "cache" in self._services:
            await cleanup_cache()

        logger.info("DI container cleaned up")

    def get_client(self, name: str):
        """Get client by name."""
        if not self._initialized:
            raise RuntimeError("DI container not initialized")
        return self._clients.get(name)

    def get_service(self, name: str):
        """Get service by name."""
        if not self._initialized:
            raise RuntimeError("DI container not initialized")
        return self._services.get(name)


# Global DI container
di_container = DIContainer()


# ========================
# Circuit Breaker (Scaffold)
# ========================

class CircuitBreaker:
    """Simple circuit breaker implementation scaffold."""

    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def call(self, func):
        """Circuit breaker decorator (simplified implementation)."""
        if not CIRCUIT_BREAKER_ENABLED:
            return func

        current_time = time.time()

        if self.state == "open":
            if current_time - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise HTTPException(status_code=503, detail="Circuit breaker is open")

        try:
            result = func()
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time

            if self.failure_count >= self.threshold:
                self.state = "open"

            raise


# Global circuit breaker
circuit_breaker = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD)


# ========================
# Application Lifespan
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with DI container management."""
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"{SERVICE_NAME} v{SERVICE_VERSION} starting up...")

    try:
        await di_container.initialize()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Application shutting down...")
    try:
        await di_container.cleanup()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# ========================
# FastAPI Application
# ========================

app = FastAPI(
    title=SERVICE_NAME,
    description="Modality-aware target scoring system with real biological data",
    version=SERVICE_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Add to logging context
    start_time = time.time()

    response = await call_next(request)

    processing_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time"] = f"{processing_time:.1f}ms"

    return response


# Simple rate limiting middleware
request_tracking = {}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    client_ip = request.client.host
    current_time = time.time()

    # Clean old requests
    if client_ip in request_tracking:
        request_tracking[client_ip] = [
            timestamp for timestamp in request_tracking[client_ip]
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
    else:
        request_tracking[client_ip] = []

    # Check rate limit
    if len(request_tracking[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "limit": RATE_LIMIT_REQUESTS, "window_seconds": RATE_LIMIT_WINDOW}
        )

    # Add current request
    request_tracking[client_ip].append(current_time)

    return await call_next(request)


# ========================
# Core API Endpoints
# ========================

@app.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck():
    """Comprehensive health check for all system components."""
    start_time = time.time()
    checks = {}
    overall_status = "healthy"

    try:
        # Check OpenTargets client
        try:
            ot_client = di_container.get_client("opentargets")
            if ot_client:
                ot_health = await ot_client.health_check()
                checks["opentargets"] = ot_health.get("status") == "healthy"
                if not checks["opentargets"]:
                    overall_status = "degraded"
            else:
                checks["opentargets"] = False
                overall_status = "degraded"
        except Exception as e:
            logger.error(f"OpenTargets health check failed: {e}")
            checks["opentargets"] = False
            overall_status = "degraded"

        # Check cache system
        try:
            cache_manager = di_container.get_service("cache")
            if cache_manager:
                cache_stats_result = await cache_stats()
                checks["cache"] = True
                checks["cache_backends"] = cache_stats_result.get("backends", {})
            else:
                checks["cache"] = False
                overall_status = "degraded"
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            checks["cache"] = False
            overall_status = "degraded"

        # Check memory usage (basic)
        import psutil
        memory_percent = psutil.virtual_memory().percent
        checks["memory_usage_percent"] = memory_percent
        if memory_percent > 90:
            overall_status = "degraded"

        # Check if any critical component failed
        if not checks.get("opentargets", False):
            overall_status = "unhealthy"

        response_time_ms = (time.time() - start_time) * 1000

        return HealthCheckResponse(
            status=overall_status,
            timestamp=time.time(),
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            checks={
                **checks,
                "response_time_ms": response_time_ms,
                "environment": {
                    "log_level": LOG_LEVEL,
                    "cache_enabled": True,
                    "circuit_breaker_enabled": CIRCUIT_BREAKER_ENABLED
                }
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=time.time(),
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            checks={"error": str(e)}
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": "Modality-aware target scoring system with real biological data",
        "phase": "1C - Data Pipeline Integration",
        "endpoints": {
            "health": "/healthcheck",
            "score": "/score",
            "sensitivity": "/sensitivity/*",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "features": [
            "Real biological data integration",
            "Async pipeline architecture",
            "Data quality validation",
            "Evidence lineage tracking",
            "Production-ready caching"
        ],
        "example_usage": {
            "curl": f"curl -X POST http://localhost:{API_PORT}/score -H 'Content-Type: application/json' -d @examples/sample_request.json"
        }
    }


@app.post("/score", response_model=ScoreResponse)
async def score_targets_endpoint(request: ScoreRequest, http_request: Request):
    """Score targets using production pipeline with data quality validation."""
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()

    # Enhanced request logging
    logger.info(
        "Scoring request received",
        extra={
            "request_id": request_id,
            "disease": request.disease,
            "target_count": len(request.targets),
            "weights": request.weights,
            "client_ip": http_request.client.host
        }
    )

    # Validate request
    is_valid, error_msg = validate_score_request(request)
    if not is_valid:
        logger.warning(
            f"Invalid request: {error_msg}",
            extra={"request_id": request_id, "validation_error": error_msg}
        )
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Circuit breaker wrapper (if enabled)
        async def _score_with_circuit_breaker():
            return await score_targets(request)

        if CIRCUIT_BREAKER_ENABLED:
            target_scores, scoring_metadata = circuit_breaker.call(lambda: asyncio.run(_score_with_circuit_breaker()))
        else:
            target_scores, scoring_metadata = await score_targets(request)

        # Defensive handling of scoring metadata
        if hasattr(scoring_metadata, 'model_dump'):
            scoring_metadata = scoring_metadata.model_dump()
        elif not isinstance(scoring_metadata, dict):
            logger.warning(f"Unexpected scoring_metadata type: {type(scoring_metadata)}")
            scoring_metadata = {
                "data_version": "Unknown",
                "meta": {
                    "cached": False,
                    "fetch_ms": 0.0,
                    "cache_hit_rate": 0.0,
                    "total_calls": len(request.targets)
                },
                "system_info": {"pipeline": "TargetBuilder"}
            }

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Safe metadata access
        meta_data = scoring_metadata.get("meta", {
            "cached": False,
            "fetch_ms": 0.0,
            "cache_hit_rate": 0.0,
            "total_calls": len(request.targets)
        })

        # Build response
        response = ScoreResponse(
            targets=target_scores,
            request_summary={
                "disease": request.disease,
                "target_count": len(request.targets),
                "weights_used": request.weights,
                "timestamp": time.time(),
                "user_id": request_id
            },
            processing_time_ms=processing_time_ms,
            data_version=scoring_metadata.get("data_version", "Unknown"),
            meta=meta_data,
            rank_impact=scoring_metadata.get("rank_impact", []),
            system_info={
                "pipeline_version": "TargetBuilder-v1.0.0-phase1c",
                "request_id": request_id,
                **scoring_metadata.get("system_info", {})
            }
        )

        # Enhanced success logging
        successful_targets = sum(1 for ts in target_scores if ts.total_score > 0.1)

        logger.info(
            "Scoring request completed successfully",
            extra={
                "request_id": request_id,
                "processing_time_ms": processing_time_ms,
                "target_count": len(target_scores),
                "successful_targets": successful_targets,
                "cache_hit_rate": meta_data.get("cache_hit_rate", 0.0),
                "data_version": response.data_version
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Scoring request failed: {e}",
            extra={
                "request_id": request_id,
                "disease": request.disease,
                "targets": request.targets,
                "error_type": type(e).__name__
            }
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during scoring: {str(e)}"
        )


@app.get("/system/info")
async def get_system_info():
    """Get comprehensive system information and configuration."""
    try:
        cache_stats_result = await cache_stats()

        return {
            "service": {
                "name": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "phase": "1C - Data Pipeline Integration"
            },
            "configuration": {
                "log_level": LOG_LEVEL,
                "log_format": LOG_FORMAT,
                "rate_limit": f"{RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s",
                "circuit_breaker_enabled": CIRCUIT_BREAKER_ENABLED,
                "cors_origins": CORS_ORIGINS
            },
            "clients": {
                "opentargets": {
                    "initialized": "opentargets" in di_container._clients,
                    "base_url": os.getenv("OT_GRAPHQL_URL", "https://api.platform.opentargets.org/api/v4/graphql")
                }
            },
            "cache": cache_stats_result,
            "scoring": {
                "default_weights": DEFAULT_WEIGHTS,
                "max_targets": 50,
                "pipeline": "TargetBuilder async"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")


@app.get("/evidence/distribution")
async def get_evidence_distribution():
    """Evidence types distribution endpoint."""
    return {
        "literature": 0,
        "databases": 0,
        "vantai": 0,
        "other": 0
    }


# ========================
# Sensitivity Router
# ========================

srouter = APIRouter(prefix="/sensitivity", tags=["sensitivity"])


def _normalize_weights(w: Dict[str, float], only: Optional[List[str]] = None) -> Dict[str, float]:
    w = dict(w or DEFAULT_WEIGHTS)
    if only:
        w = {k: v for k, v in w.items() if k in only}
    w = {k: float(v) for k, v in w.items() if float(v) > 0}
    if not w:
        keys = only or list(DEFAULT_WEIGHTS.keys())
        return {k: 1.0 / len(keys) for k in keys}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


async def _score_many(req: ScoreRequest) -> Dict[str, float]:
    target_scores, _ = await score_targets(req)
    return {ts.target: float(ts.total_score or 0.0) for ts in target_scores}


def _dirichlet_around(weights: Dict[str, float], alpha: float) -> Dict[str, float]:
    keys = list(weights.keys())
    base = np.array([max(1e-9, weights[k]) for k in keys], dtype=float)
    base = base / base.sum()
    sample = np.random.dirichlet(base * float(alpha))
    return {k: float(v) for k, v in zip(keys, sample)}


# ---- Ablation ----
@srouter.post("/ablation")
async def ablation_post(req: ScoreRequest, http_request: Request):
    rid = getattr(http_request.state, 'request_id', str(uuid.uuid4()))
    try:
        base_w = _normalize_weights(req.weights or DEFAULT_WEIGHTS)
        baseline = await _score_many(ScoreRequest(disease=req.disease, targets=req.targets, weights=base_w))
        channels = list(base_w.keys())
        results = []
        for gene in req.targets:
            entry = {"target": gene, "baseline": baseline.get(gene, 0.0), "ablations": []}
            for ch in channels:
                w_drop = _normalize_weights({k: (v if k != ch else 0.0) for k, v in base_w.items()}, only=channels)
                ch_score = (await _score_many(ScoreRequest(disease=req.disease, targets=[gene], weights=w_drop)))[gene]
                entry["ablations"].append({"channel": ch, "score": ch_score, "delta": ch_score - entry["baseline"]})
            results.append(entry)
        return {"disease": req.disease, "weights_used": base_w, "targets": results, "request_id": rid}
    except Exception as e:
        logger.error(f"Ablation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@srouter.get("/ablation")
async def ablation_get(disease: str, targets: str, base_weights: Optional[str] = None):
    weights = DEFAULT_WEIGHTS if not base_weights else {**DEFAULT_WEIGHTS, **json.loads(base_weights)}
    req = ScoreRequest(disease=disease,
                       targets=[t.strip().upper() for t in targets.split(",") if t.strip()],
                       weights=weights)
    return await ablation_post(req, Request)  # type: ignore


# ---- Weight Impact ----
@srouter.post("/weight-impact")
async def weight_impact_post(req: ScoreRequest):
    try:
        base_w = _normalize_weights(req.weights or DEFAULT_WEIGHTS)
        baseline = await _score_many(ScoreRequest(disease=req.disease, targets=req.targets, weights=base_w))
        channels = list(base_w.keys())
        PCT = 0.10

        def bump(weights: Dict[str, float], key: str, pct: float) -> Dict[str, float]:
            w = dict(weights)
            w[key] = max(1e-9, w[key] * (1.0 + pct))
            return _normalize_weights(w, only=list(weights.keys()))

        results = []
        for gene in req.targets:
            impacts = []
            for ch in channels:
                up_w = bump(base_w, ch, +PCT)
                dn_w = bump(base_w, ch, -PCT)
                up_score = (await _score_many(ScoreRequest(disease=req.disease, targets=[gene], weights=up_w)))[gene]
                down_score = (await _score_many(ScoreRequest(disease=req.disease, targets=[gene], weights=dn_w)))[gene]
                impacts.append({
                    "channel": ch, "baseline": baseline[gene],
                    "up_score": up_score, "down_score": down_score,
                    "up_delta": up_score - baseline[gene], "down_delta": down_score - baseline[gene]
                })
            results.append({"target": gene, "impacts": impacts})
        return {"disease": req.disease, "weights_used": base_w, "targets": results}
    except Exception as e:
        logger.error(f"Weight impact analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@srouter.get("/weight-impact")
async def weight_impact_get(disease: str, targets: str, base_weights: Optional[str] = None):
    weights = DEFAULT_WEIGHTS if not base_weights else {**DEFAULT_WEIGHTS, **json.loads(base_weights)}
    req = ScoreRequest(disease=disease,
                       targets=[t.strip().upper() for t in targets.split(",") if t.strip()],
                       weights=weights)
    return await weight_impact_post(req)


# ---- Stability / Simulate ----
@srouter.post("/stability")
async def stability_post(req: ScoreRequest, samples: int = 200, alpha: float = 80.0):
    baseline_scores, _ = await score_targets(req)
    baseline_ranks = {ts.target: i+1 for i, ts in enumerate(sorted(baseline_scores, key=lambda x: -x.total_score))}
    per_target_ranks: Dict[str, List[int]] = {ts.target: [] for ts in baseline_scores}
    for _ in range(samples):
        w = _dirichlet_around(req.weights or DEFAULT_WEIGHTS, alpha)
        sim_scores, _ = await score_targets(ScoreRequest(disease=req.disease, targets=req.targets, weights=w))
        sim_sorted = sorted(sim_scores, key=lambda x: -x.total_score)
        for i, ts in enumerate(sim_sorted):
            per_target_ranks[ts.target].append(i+1)
    out = []
    for t in req.targets:
        ranks = per_target_ranks.get(t, []) or [baseline_ranks.get(t, len(req.targets))]
        out.append({
            "target": t, "baseline_rank": baseline_ranks.get(t, None),
            "mean_rank": float(np.mean(ranks)), "std_rank": float(np.std(ranks)),
            "best_rank": int(np.min(ranks)), "worst_rank": int(np.max(ranks)), "n": len(ranks)
        })
    return {"samples": samples, "alpha": alpha, "stability": sorted(out, key=lambda r: r["mean_rank"])}


@srouter.get("/stability")
async def stability_get(disease: str, targets: str, samples: int = 200, alpha: float = 80.0):
    req = ScoreRequest(disease=disease,
                       targets=[t.strip().upper() for t in targets.split(",") if t.strip()],
                       weights=DEFAULT_WEIGHTS)
    return await stability_post(req, samples=samples, alpha=alpha)


# Include sensitivity router
app.include_router(srouter)

# FE /api öneki için aynısını alias olarak ekle
app.include_router(srouter, prefix="/api")


@app.get("/debug/opentargets")
async def debug_opentargets():
    """Debug OpenTargets client status."""
    try:
        ot_client = await get_ot_client()
        health = await ot_client.health_check()

        # Test a known association
        test_associations = await ot_client.fetch_gene_disease_associations(
            "BRCA1", "EFO_0000305"
        )

        return {
            "client_health": health,
            "demo_mode": health.get("mode") == "demo",
            "test_association_count": len(test_associations) if test_associations else 0,
            "test_associations": test_associations[:2] if test_associations else []
        }
    except Exception as e:
        return {"error": str(e)}
# ========================
# Global Exception Handler
# ========================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    logger.error(
        f"Unhandled exception: {exc}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "exception_type": type(exc).__name__
        }
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__},
            timestamp=time.time(),
            request_id=request_id
        ).model_dump()
    )


# ========================
# Dependency injection functions
# ========================

async def get_ot_client_dependency():
    """FastAPI dependency for OpenTargets client."""
    return di_container.get_client("opentargets")


async def get_cache_dependency():
    """FastAPI dependency for cache manager."""
    return di_container.get_service("cache")


# ========================
# Environment configuration documentation
# ========================

def get_env_config_docs() -> Dict[str, str]:
    """Document all configurable environment variables."""
    return {
        # Service config
        "SERVICE_NAME": "Service name for logging and health checks",
        "SERVICE_VERSION": "Service version string",
        "API_HOST": "API bind host (default: 0.0.0.0)",
        "API_PORT": "API bind port (default: 8001)",

        # External APIs
        "OT_GRAPHQL_URL": "OpenTargets GraphQL endpoint",
        "OT_REST_URL": "OpenTargets REST endpoint",
        "OT_MAX_RETRIES": "Max retries for OpenTargets API",
        "OT_TIMEOUT_SECONDS": "Request timeout for OpenTargets",

        # Cache configuration
        "CACHE_DIR": "Directory for file cache",
        "REDIS_URL": "Redis connection URL (optional)",
        "CACHE_NAMESPACE": "Cache key namespace",

        # Scoring weights
        "WEIGHT_GENETICS": "Genetics channel weight (default: 0.35)",
        "WEIGHT_PPI": "PPI proximity weight (default: 0.25)",
        "WEIGHT_PATHWAY": "Pathway enrichment weight (default: 0.20)",
        "WEIGHT_SAFETY": "Safety penalty weight (default: 0.10)",
        "WEIGHT_MODALITY": "Modality fit weight (default: 0.10)",

        # Performance
        "RATE_LIMIT_REQUESTS": "Requests per window (default: 100)",
        "RATE_LIMIT_WINDOW": "Rate limit window in seconds (default: 3600)",
        "CIRCUIT_BREAKER_ENABLED": "Enable circuit breaker (default: false)",
        "CIRCUIT_BREAKER_THRESHOLD": "Circuit breaker failure threshold",

        # Logging
        "LOG_LEVEL": "Logging level (DEBUG, INFO, WARNING, ERROR)",
        "LOG_FORMAT": "Log format (json, text)",

        # Data quality
        "OT_STALENESS_HOURS": "OpenTargets data staleness threshold",
        "STRING_STALENESS_HOURS": "STRING data staleness threshold",
        "DEFAULT_STALENESS_HOURS": "Default staleness threshold"
    }


@app.get("/debug/ot-query")
async def debug_ot_query():
    """Debug actual OpenTargets queries."""
    try:
        ot_client = await get_ot_client()

        # Test with known gene-disease pair
        gene = "EGFR"
        disease = "EFO_0003071"

        # Get the actual query being sent
        associations = await ot_client.fetch_gene_disease_associations(gene, disease)

        return {
            "gene_tested": gene,
            "disease_tested": disease,
            "associations_count": len(associations) if associations else 0,
            "first_association": associations[0].__dict__ if associations else None,
            "client_config": {
                "graphql_url": getattr(ot_client, 'graphql_url', 'unknown'),
                "rest_url": getattr(ot_client, 'rest_url', 'unknown'),
            },
            "raw_response_preview": str(associations)[:500] if associations else "No data"
        }
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}
@app.get("/debug/ot-status")
async def debug_ot_status():
    """Debug OpenTargets client and test known associations."""
    try:
        ot_client = await get_ot_client()
        health = await ot_client.health_check()

        # Test known strong association
        brca_breast = await ot_client.fetch_gene_disease_associations("BRCA1", "EFO_0000305")

        return {
            "opentargets_health": health,
            "demo_mode": health.get("mode") == "demo",
            "test_brca1_breast": {
                "association_count": len(brca_breast) if brca_breast else 0,
                "first_association": brca_breast[0].__dict__ if brca_breast else None
            },
            "base_url": os.getenv("OT_GRAPHQL_URL", "default"),
            "client_type": type(ot_client).__name__
        }
    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}


@app.get("/debug/genetics-location/{gene}")
async def debug_genetics_location(gene: str):
    """Debug where genetics scores are actually stored."""
    try:
        # Score with valid disease ID
        req = ScoreRequest(
            disease="EFO_0000692",  # Use valid lung carcinoma ID
            targets=[gene],
            weights=DEFAULT_WEIGHTS
        )
        target_scores, metadata = await score_targets(req)

        if target_scores:
            ts = target_scores[0]
            return {
                "target": gene,
                "total_score": ts.total_score,
                "raw_data": ts.__dict__ if hasattr(ts, '__dict__') else str(ts),
                "genetics_paths": {
                    "breakdown.genetics": getattr(ts, 'breakdown', {}).get('genetics') if hasattr(ts,
                                                                                                  'breakdown') else 'no_breakdown',
                    "channels.genetics.score": getattr(ts, 'channels', {}).get('genetics', {}).get('score') if hasattr(
                        ts, 'channels') else 'no_channels',
                    "explanation.contributions": [c for c in getattr(ts, 'explanation', {}).get('contributions', []) if
                                                  c.get('channel') == 'genetics'] if hasattr(ts,
                                                                                             'explanation') else 'no_explanation'
                }
            }
        else:
            return {"error": "No target scores returned"}

    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )