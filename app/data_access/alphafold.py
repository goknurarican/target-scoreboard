# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
AlphaFold API client for protein structure confidence data.
Production-ready with confidence score extraction and quality validation.
"""
import asyncio
import logging
import os
import time
from typing import Dict, Optional, Any
from urllib.parse import quote

import httpx

from ..schemas import StructureConfidence, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from .cache import cached_fetch

# Configure logger
logger = logging.getLogger(__name__)

# Configuration from environment
ALPHAFOLD_API_URL = os.getenv("ALPHAFOLD_API_URL", "https://alphafold.ebi.ac.uk/api")
ALPHAFOLD_VERSION = os.getenv("ALPHAFOLD_VERSION", "v4")
REQUEST_TIMEOUT = int(os.getenv("ALPHAFOLD_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("ALPHAFOLD_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("ALPHAFOLD_RETRY_BACKOFF", "1.0"))

# Quality thresholds
HIGH_CONFIDENCE_PLDDT = float(os.getenv("HIGH_CONFIDENCE_PLDDT", "90.0"))
MEDIUM_CONFIDENCE_PLDDT = float(os.getenv("MEDIUM_CONFIDENCE_PLDDT", "70.0"))
MAX_PAE_THRESHOLD = float(os.getenv("MAX_PAE_THRESHOLD", "5.0"))


class AlphaFoldError(Exception):
    """AlphaFold specific errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class AlphaFoldClient:
    """
    Production AlphaFold API client for structure confidence data.

    Features:
    - pLDDT and PAE confidence extraction
    - Structure quality assessment
    - UniProt ID resolution
    - Cache integration
    - Quality flags for low-confidence regions
    """

    def __init__(self, base_url: str = ALPHAFOLD_API_URL):
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

    async def _make_request(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Dict:
        """
        Make HTTP request to AlphaFold API with retries.

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
                    logger.warning(f"AlphaFold rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle errors
                if response.status_code >= 500:
                    raise AlphaFoldError(f"AlphaFold server error: {response.status_code}", response.status_code)
                elif response.status_code == 404:
                    # Structure not found - return None rather than error
                    logger.info(f"Structure not found in AlphaFold: {endpoint}")
                    return {"empty_result": True}
                elif response.status_code >= 400:
                    raise AlphaFoldError(f"AlphaFold client error: {response.status_code}", response.status_code)

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"AlphaFold timeout, retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES and isinstance(e, (httpx.NetworkError, AlphaFoldError)):
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"AlphaFold error, retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # All retries failed
        error_msg = f"AlphaFold API failed after {MAX_RETRIES} retries: {last_exception}"
        logger.error(error_msg, extra={"endpoint": endpoint, "params": params})
        raise AlphaFoldError(error_msg) from last_exception

    async def _resolve_uniprot_id(self, gene: str) -> Optional[str]:
        """
        Resolve gene symbol to UniProt ID for AlphaFold lookup.

        Args:
            gene: Gene symbol

        Returns:
            UniProt ID or None if not found
        """
        # Simple mapping for common genes (expand as needed)
        gene_to_uniprot = {
            "EGFR": "P00533",
            "ERBB2": "P04626",
            "MET": "P08581",
            "ALK": "Q9UM73",
            "KRAS": "P01116",
            "BRAF": "P15056",
            "PIK3CA": "P42336",
            "ROS1": "P08922",
            "RET": "P07949",
            "NTRK1": "P04629",
            "TP53": "P04637",
            "BRCA1": "P38398",
            "BRCA2": "P51587",
            "PTEN": "P60484",
            "VHL": "P40337"
        }

        uniprot_id = gene_to_uniprot.get(gene.upper())
        if uniprot_id:
            return uniprot_id

        # TODO: Implement UniProt API lookup for unmapped genes
        logger.warning(f"UniProt ID not found for gene {gene} (limited mapping available)")
        return None

    async def get_confidence_scores(self, gene: str) -> Optional[StructureConfidence]:
        """
        Get structure confidence scores for a gene from AlphaFold.
        """

        async def _fetch_confidence():
            # Resolve to UniProt ID
            uniprot_id = await self._resolve_uniprot_id(gene)
            if not uniprot_id:
                logger.info(f"Cannot resolve {gene} to UniProt ID for AlphaFold lookup")
                return None

            try:
                # Fetch structure summary
                response_data = await self._make_request(f"prediction/{uniprot_id}")

                if response_data.get("empty_result"):
                    logger.info(f"No AlphaFold structure for {gene} ({uniprot_id})")
                    return None

                # Extract confidence scores
                plddt_data = response_data.get("plddt", [])
                pae_data = response_data.get("pae", [])

                # Calculate mean pLDDT - GÜVENLI PARSING
                plddt_mean = None
                if plddt_data:
                    try:
                        if isinstance(plddt_data, list) and len(plddt_data) > 0:
                            plddt_values = []
                            for x in plddt_data:
                                if isinstance(x, (int, float)):
                                    plddt_values.append(float(x))

                            if plddt_values:
                                plddt_mean = sum(plddt_values) / len(plddt_values)
                    except Exception as e:
                        logger.warning(f"pLDDT parsing error for {gene}: {e}")

                # Calculate mean PAE - GÜVENLI PARSING
                pae_mean = None
                if pae_data:
                    try:
                        if isinstance(pae_data, list) and len(pae_data) > 0:
                            flat_pae = []
                            for row in pae_data:
                                if isinstance(row, list):
                                    for x in row:
                                        if isinstance(x, (int, float)):
                                            flat_pae.append(float(x))
                                elif isinstance(row, (int, float)):
                                    flat_pae.append(float(row))

                            if flat_pae:
                                pae_mean = sum(flat_pae) / len(flat_pae)
                    except Exception as e:
                        logger.warning(f"PAE parsing error for {gene}: {e}")

                # Create confidence record
                confidence = StructureConfidence(
                    gene=gene.upper(),
                    plddt_mean=plddt_mean,
                    pae_mean=pae_mean,
                    timestamp=get_utc_now(),
                    source=f"alphafold_{ALPHAFOLD_VERSION}"
                )

                return confidence

            except AlphaFoldError as e:
                if e.status_code == 404:
                    logger.info(f"AlphaFold structure not available for {gene} ({uniprot_id})")
                    return None
                else:
                    raise
            except Exception as e:
                logger.error(f"AlphaFold data processing error for {gene}: {e}")
                return None

        try:
            return await cached_fetch(
                source="alphafold",
                method="get_confidence_scores",
                fetch_coro=_fetch_confidence,
                gene=gene
            )
        except Exception as e:
            logger.error(f"AlphaFold cache fetch error for {gene}: {e}")
            # Return None instead of raising - let modality channel handle missing data
            return None
    def _assess_confidence_level(self, plddt_mean: Optional[float], pae_mean: Optional[float]) -> str:
        """
        Assess overall structure confidence level.

        Args:
            plddt_mean: Mean pLDDT score
            pae_mean: Mean PAE score

        Returns:
            Confidence level string
        """
        if plddt_mean is None:
            return "unknown"

        if plddt_mean >= HIGH_CONFIDENCE_PLDDT:
            if pae_mean is not None and pae_mean <= MAX_PAE_THRESHOLD:
                return "very_high"
            return "high"
        elif plddt_mean >= MEDIUM_CONFIDENCE_PLDDT:
            return "medium"
        else:
            return "low"

    async def get_structure_summary(self, gene: str) -> Dict[str, Any]:
        """
        Get basic structure information summary.

        Args:
            gene: Gene symbol

        Returns:
            Dict with structure summary or empty dict if not available
        """

        async def _fetch_summary():
            uniprot_id = await self._resolve_uniprot_id(gene)
            if not uniprot_id:
                return {}

            try:
                response_data = await self._make_request(f"prediction/{uniprot_id}/summary")

                if response_data.get("empty_result"):
                    return {}

                # Extract basic info
                summary = {
                    "uniprot_id": uniprot_id,
                    "gene_symbol": gene.upper(),
                    "model_version": response_data.get("modelVersion", "unknown"),
                    "sequence_length": response_data.get("sequenceLength", 0),
                    "coverage": response_data.get("coverage", {}),
                    "url": f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}"
                }

                return summary

            except Exception as e:
                logger.error(f"AlphaFold summary fetch failed for {gene}: {e}")
                return {}

        return await cached_fetch(
            source="alphafold",
            method="get_structure_summary",
            fetch_coro=_fetch_summary,
            gene=gene
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check AlphaFold API health.

        Returns:
            Health status dict
        """
        try:
            start_time = time.time()

            # Test with known UniProt ID
            response_data = await self._make_request("prediction/P00533/summary")  # EGFR

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "api_version": ALPHAFOLD_VERSION,
                "high_confidence_threshold": HIGH_CONFIDENCE_PLDDT,
                "timestamp": get_utc_now().isoformat()
            }

        except Exception as e:
            logger.error(f"AlphaFold health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": get_utc_now().isoformat()
            }


# ========================
# Global client management
# ========================

_alphafold_client: Optional[AlphaFoldClient] = None


async def get_alphafold_client() -> AlphaFoldClient:
    """Get global AlphaFold client instance."""
    global _alphafold_client
    if _alphafold_client is None:
        _alphafold_client = AlphaFoldClient()
        logger.info("AlphaFold client initialized")
    return _alphafold_client


async def cleanup_alphafold_client() -> None:
    """Cleanup global AlphaFold client."""
    global _alphafold_client
    if _alphafold_client is not None:
        await _alphafold_client.close()
        _alphafold_client = None
        logger.info("AlphaFold client cleaned up")