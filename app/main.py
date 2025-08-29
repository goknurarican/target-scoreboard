# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
VantAI Target Scoreboard FastAPI Application - Phase 1C Production.
Features: DI wiring, structured logging, healthcheck, circuit breaker scaffold.
"""
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .schemas import (
    ScoreRequest,
    ScoreResponse,
    HealthCheckResponse,
    ErrorResponse
)
from .scoring import score_targets, validate_score_request
from .data_access.opentargets import get_ot_client, cleanup_ot_client
from .data_access.cache import get_cache_manager, cleanup_cache, cache_stats
logger = logging.getLogger(__name__)

# ========================
# Configuration from Environment
# ========================

# Service configuration
SERVICE_NAME = os.getenv("SERVICE_NAME", "VantAI Target Scoreboard")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0-phase1c")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

# Circuit breaker configuration
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "false").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # json or text


# ========================
# Structured Logging Setup
# ========================

def setup_logging():
    """Configure structured logging with request IDs."""
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    if LOG_FORMAT == "json":
        # JSON formatter for production
        import json

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
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
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

            # TODO: Initialize other clients in Phase 1B
            # self._clients["stringdb"] = await get_stringdb_client()
            # self._clients["expression_atlas"] = await get_expression_atlas_client()
            # self._clients["alphafold"] = await get_alphafold_client()
            # self._clients["pubmed"] = await get_pubmed_client()

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
# Request ID Middleware
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
# API Endpoints
# ========================

@app.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck():
    """
    Comprehensive health check for all system components.

    Returns:
        HealthCheckResponse with detailed component status
    """
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

    logger = logging.getLogger(__name__)

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

        # ✅ DEFENSIVE HANDLING OF CacheMetadata OBJECT
        if hasattr(scoring_metadata, 'model_dump'):
            # If it's a Pydantic object, convert to dict
            scoring_metadata = scoring_metadata.model_dump()
        elif not isinstance(scoring_metadata, dict):
            # If it's not a dict, create a fallback dict
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

        # ✅ SAFE METADATA ACCESS
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
            meta=meta_data,  # ✅ Safe metadata
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
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log and convert to HTTP exception
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
                # TODO: Add other clients in Phase 1B
            },
            "cache": cache_stats_result,
            "scoring": {
                "default_weights": {
                    "genetics": float(os.getenv("WEIGHT_GENETICS", "0.35")),
                    "ppi": float(os.getenv("WEIGHT_PPI", "0.25")),
                    "pathway": float(os.getenv("WEIGHT_PATHWAY", "0.20")),
                    "safety": float(os.getenv("WEIGHT_SAFETY", "0.10")),
                    "modality_fit": float(os.getenv("WEIGHT_MODALITY", "0.10"))
                },
                "max_targets": 50,
                "pipeline": "TargetBuilder async"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")


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


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    )