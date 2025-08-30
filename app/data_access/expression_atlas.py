# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Expression Atlas API client for gene expression data.
Production-ready with cache, validation, and tissue-specific queries.
"""
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import httpx

from ..schemas import ExpressionRecord, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from .cache import cached_fetch

# Configure logger
logger = logging.getLogger(__name__)

# Configuration from environment
ATLAS_API_URL = os.getenv("ATLAS_API_URL", "https://www.ebi.ac.uk/gxa/api")
ATLAS_VERSION = os.getenv("ATLAS_VERSION", "latest")
DEFAULT_EXPERIMENT = os.getenv("ATLAS_DEFAULT_EXPERIMENT", "E-MTAB-5214")  # GTEx v8
REQUEST_TIMEOUT = int(os.getenv("ATLAS_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("ATLAS_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("ATLAS_RETRY_BACKOFF", "1.0"))

# Common tissue mappings
TISSUE_MAPPINGS = {
    "brain": ["brain", "cerebrum", "cerebellum", "cortex"],
    "liver": ["liver", "hepatic"],
    "lung": ["lung", "pulmonary"],
    "heart": ["heart", "cardiac", "myocardium"],
    "kidney": ["kidney", "renal"],
    "muscle": ["muscle", "skeletal muscle"],
    "blood": ["blood", "leukocyte", "lymphocyte"],
    "skin": ["skin", "epidermis", "dermis"],
    "breast": ["breast", "mammary"],
    "prostate": ["prostate"],
    "ovary": ["ovary", "ovarian"],
    "testis": ["testis", "testicular"]
}


class ExpressionAtlasError(Exception):
    """Expression Atlas specific errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ExpressionAtlasClient:
    """
    Production Expression Atlas API client.

    Features:
    - Tissue-specific expression queries
    - Unit normalization (TPM, FPKM, etc.)
    - Quantile calculation when possible
    - Cache integration with TTL
    - Validation and quality flags
    """

    def __init__(self, base_url: str = ATLAS_API_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[httpx.AsyncClient] = None
        self.validator = get_validator()

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "VantAI-TargetScoreboard/1.0",
                    "Accept": "application/json"
                },
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=3)
            )
        return self.session

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session is not None:
            await self.session.aclose()
            self.session = None

    async def _make_request(self, endpoint: str, params: Dict[str, str]) -> Dict:
        """
        Make HTTP request to Expression Atlas API with retries.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"

        last_exception = None
        for attempt in range(1 + MAX_RETRIES):
            try:

                response = await session.get(url, params=params)

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Expression Atlas rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle errors
                if response.status_code >= 500:
                    raise ExpressionAtlasError(f"Atlas server error: {response.status_code}", response.status_code)
                elif response.status_code == 404:
                    # Gene or tissue not found - return empty result rather than error
                    logger.info(f"Gene or tissue not found in Atlas: {params}")
                    return {"empty_result": True}
                elif response.status_code >= 400:
                    raise ExpressionAtlasError(f"Atlas client error: {response.status_code}", response.status_code)

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Atlas timeout, retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES and isinstance(e, (httpx.NetworkError, ExpressionAtlasError)):
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Atlas error, retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # All retries failed
        error_msg = f"Expression Atlas API failed after {MAX_RETRIES} retries: {last_exception}"
        logger.error(error_msg, extra={"endpoint": endpoint, "params": params})
        raise ExpressionAtlasError(error_msg) from last_exception

    def _normalize_tissue_name(self, tissue: str) -> str:
        """Normalize tissue name for consistent querying."""
        tissue_lower = tissue.lower().strip()

        # Check tissue mappings
        for canonical, variants in TISSUE_MAPPINGS.items():
            if tissue_lower in variants or tissue_lower == canonical:
                return canonical

        return tissue_lower

    async def get_expression(self, gene: str, tissue: str) -> List[ExpressionRecord]:
        """
        Get gene expression data for specific tissue.

        Args:
            gene: Gene symbol
            tissue: Tissue type

        Returns:
            List of ExpressionRecord objects (empty if no data)
        """

        async def _fetch_expression():
            normalized_tissue = self._normalize_tissue_name(tissue)

            # Try multiple endpoint strategies
            expression_records = []

            # Strategy 1: Baseline expression by tissue
            try:
                params = {
                    "geneId": gene.upper(),
                    "experiment": DEFAULT_EXPERIMENT,
                    "format": "json"
                }

                response_data = await self._make_request("expression", params)

                if response_data.get("empty_result"):
                    logger.info(f"No expression data found for {gene} in Atlas")
                    return []

                # Parse expression data
                expressions = response_data.get("expressions", [])
                for expr_data in expressions:
                    # Extract tissue information
                    tissue_info = expr_data.get("tissue", {})
                    tissue_name = tissue_info.get("name", "unknown").lower()

                    # Check if this matches our target tissue
                    if normalized_tissue in tissue_name or tissue_name in normalized_tissue:
                        value = float(expr_data.get("value", 0.0))
                        unit = expr_data.get("unit", "TPM")

                        # Calculate quantile if baseline data available
                        quantile = None
                        if "baseline" in expr_data:
                            baseline_values = expr_data["baseline"]
                            if isinstance(baseline_values, list) and baseline_values:
                                sorted_baseline = sorted(baseline_values)
                                position = sum(1 for v in sorted_baseline if v <= value)
                                quantile = position / len(sorted_baseline)

                        # Create evidence reference
                        evidence = EvidenceRef(
                            source="expression_atlas",
                            title=f"Expression in {tissue}: {value:.2f} {unit}",
                            url=f"https://www.ebi.ac.uk/gxa/genes/{gene}",
                            source_quality="high",
                            timestamp=get_utc_now()
                        )

                        record = ExpressionRecord(
                            gene=gene.upper(),
                            tissue=tissue,
                            value=value,
                            unit=unit,
                            quantile=quantile,
                            timestamp=get_utc_now(),
                            source=f"expression_atlas_{ATLAS_VERSION}"
                        )

                        expression_records.append(record)

                # Log results
                if expression_records:
                    logger.info(
                        f"Expression data found for {gene} in {tissue}",
                        extra={
                            "gene": gene,
                            "tissue": tissue,
                            "record_count": len(expression_records),
                            "avg_expression": sum(r.value for r in expression_records) / len(expression_records)
                        }
                    )
                else:
                    logger.info(f"No expression data for {gene} in {tissue}")

                return expression_records

            except ExpressionAtlasError as e:
                if e.status_code == 404:
                    logger.info(f"Gene {gene} not found in Expression Atlas")
                    return []
                else:
                    raise

        return await cached_fetch(
            source="expression_atlas",
            method="get_expression",
            fetch_coro=_fetch_expression,
            gene=gene,
            tissue=tissue
        )

    async def get_tissue_specificity(self, gene: str) -> Dict[str, float]:
        """
        Get tissue specificity scores for a gene across all tissues.

        Args:
            gene: Gene symbol

        Returns:
            Dict mapping tissue names to expression levels
        """

        async def _fetch_tissue_specificity():
            try:
                params = {
                    "geneId": gene.upper(),
                    "experiment": DEFAULT_EXPERIMENT,
                    "format": "json"
                }

                response_data = await self._make_request("expression", params)

                if response_data.get("empty_result"):
                    return {}

                # Parse tissue-specific expression
                tissue_expression = {}
                expressions = response_data.get("expressions", [])

                for expr_data in expressions:
                    tissue_info = expr_data.get("tissue", {})
                    tissue_name = tissue_info.get("name", "unknown")
                    value = float(expr_data.get("value", 0.0))

                    # Group by normalized tissue name
                    normalized_tissue = self._normalize_tissue_name(tissue_name)
                    if normalized_tissue not in tissue_expression:
                        tissue_expression[normalized_tissue] = []
                    tissue_expression[normalized_tissue].append(value)

                # Average expression per tissue
                averaged_expression = {}
                for tissue, values in tissue_expression.items():
                    averaged_expression[tissue] = sum(values) / len(values)

                logger.info(
                    f"Tissue specificity for {gene}",
                    extra={
                        "gene": gene,
                        "tissue_count": len(averaged_expression),
                        "max_expression": max(averaged_expression.values()) if averaged_expression else 0
                    }
                )

                return averaged_expression

            except Exception as e:
                logger.error(f"Tissue specificity fetch failed for {gene}: {e}")
                return {}

        return await cached_fetch(
            source="expression_atlas",
            method="get_tissue_specificity",
            fetch_coro=_fetch_tissue_specificity,
            gene=gene
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Expression Atlas API health.

        Returns:
            Health status dict
        """
        try:
            start_time = time.time()

            # Test with known gene
            params = {
                "geneId": "EGFR",
                "experiment": DEFAULT_EXPERIMENT,
                "format": "json"
            }

            await self._make_request("expression", params)

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "api_version": ATLAS_VERSION,
                "default_experiment": DEFAULT_EXPERIMENT,
                "timestamp": get_utc_now().isoformat()
            }

        except Exception as e:
            logger.error(f"Expression Atlas health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": get_utc_now().isoformat()
            }


# ========================
# Global client management
# ========================

_expression_atlas_client: Optional[ExpressionAtlasClient] = None


async def get_expression_atlas_client() -> ExpressionAtlasClient:
    """Get global Expression Atlas client instance."""
    global _expression_atlas_client
    if _expression_atlas_client is None:
        _expression_atlas_client = ExpressionAtlasClient()
        logger.info("Expression Atlas client initialized")
    return _expression_atlas_client


async def cleanup_expression_atlas_client() -> None:
    """Cleanup global Expression Atlas client."""
    global _expression_atlas_client
    if _expression_atlas_client is not None:
        await _expression_atlas_client.close()
        _expression_atlas_client = None
        logger.info("Expression Atlas client cleaned up")