# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
STRING-DB API client for protein-protein interaction data.
Production-ready with cache, backoff, validation, and structured logging.
"""
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import httpx

from ..schemas import PPINetwork, PPIEdge, EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from .cache import cached_fetch

# Configure logger
logger = logging.getLogger(__name__)

# Configuration from environment
STRING_API_URL = os.getenv("STRING_API_URL", "https://string-db.org/api")
STRING_VERSION = os.getenv("STRING_VERSION", "12.0")
DEFAULT_SPECIES = int(os.getenv("STRING_SPECIES", "9606"))  # Human
MIN_CONFIDENCE = float(os.getenv("STRING_MIN_CONFIDENCE", "0.4"))
MAX_INTERACTIONS = int(os.getenv("STRING_MAX_INTERACTIONS", "50"))
REQUEST_TIMEOUT = int(os.getenv("STRING_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("STRING_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("STRING_RETRY_BACKOFF", "1.0"))


class StringDBError(Exception):
    """STRING-DB specific errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class StringDBClient:
    """
    Production STRING-DB API client.

    Features:
    - Async HTTP with retry and backoff
    - Response validation and quality flags
    - Structured error logging and monitoring
    - Production cache integration
    - Configurable confidence thresholds
    """

    def __init__(self, base_url: str = STRING_API_URL):
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
        Make HTTP request to STRING API with retries.

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
                    logger.warning(f"STRING API rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle errors
                if response.status_code >= 500:
                    raise StringDBError(f"STRING server error: {response.status_code}", response.status_code)
                elif response.status_code >= 400:
                    raise StringDBError(f"STRING client error: {response.status_code}", response.status_code)

                response.raise_for_status()

                # Parse response
                if response.headers.get("content-type", "").startswith("application/json"):
                    return response.json()
                else:
                    # Handle TSV response
                    return {"text_data": response.text}

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"STRING timeout, retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES and isinstance(e, (httpx.NetworkError, StringDBError)):
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"STRING error, retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # All retries failed
        error_msg = f"STRING API failed after {MAX_RETRIES} retries: {last_exception}"
        logger.error(error_msg, extra={"endpoint": endpoint, "params": params})
        raise StringDBError(error_msg) from last_exception

    async def fetch_ppi(
            self,
            gene: str,
            species: Optional[int] = None,
            limit: int = None
    ) -> PPINetwork:
        """
        Fetch protein-protein interactions for a gene from STRING-DB.

        Args:
            gene: Gene symbol or protein identifier
            species: NCBI taxonomy ID (default: 9606 for human)
            limit: Maximum number of interactions (default: MAX_INTERACTIONS)

        Returns:
            PPINetwork object with edges and metadata
        """

        async def _fetch_ppi():
            # Configuration
            species_id = species or DEFAULT_SPECIES
            interaction_limit = min(limit or MAX_INTERACTIONS, MAX_INTERACTIONS)

            # Build request parameters
            params = {
                "identifiers": gene,
                "species": str(species_id),
                "required_score": str(int(MIN_CONFIDENCE * 1000)),  # STRING uses 0-1000 scale
                "network_type": "functional",  # functional, physical, or full
                "limit": str(interaction_limit),
                "caller_identity": "vantai.target.scoreboard"
            }

            # Fetch interaction data
            start_time = time.time()
            response_data = await self._make_request("tsv/interaction_partners", params)
            fetch_time = (time.time() - start_time) * 1000

            # Parse TSV response
            edges = []
            if "text_data" in response_data:
                lines = response_data["text_data"].strip().split('\n')

                # Skip header if present
                data_lines = lines[1:] if lines and lines[0].startswith('#') or 'stringId' in lines[0] else lines

                for line in data_lines:
                    if not line.strip():
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 3:  # stringId_A, stringId_B, score
                        try:
                            partner_id = parts[1]
                            confidence_score = float(parts[2]) / 1000.0  # Convert from 0-1000 to 0-1

                            # Extract gene symbol from STRING ID (format: species.gene)
                            partner_gene = partner_id.split('.')[-1] if '.' in partner_id else partner_id

                            # Create evidence reference
                            evidence = EvidenceRef(
                                source="stringdb",
                                title=f"STRING interaction: {gene} - {partner_gene}",
                                url=f"https://string-db.org/network/{gene}",
                                source_quality="high" if confidence_score >= 0.7 else "medium",
                                timestamp=get_utc_now()
                            )

                            edge = PPIEdge(
                                source_gene=gene.upper(),
                                partner=partner_gene.upper(),
                                confidence=confidence_score,
                                evidence=[evidence]
                            )

                            edges.append(edge)

                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing STRING line '{line}': {e}")
                            continue

            # Create network object
            network = PPINetwork(
                gene=gene.upper(),
                edges=edges,
                timestamp=get_utc_now(),
                source=f"stringdb_v{STRING_VERSION}"
            )

            # Log network quality metrics
            if edges:
                avg_confidence = sum(edge.confidence for edge in edges) / len(edges)
                high_conf_count = sum(1 for edge in edges if edge.confidence >= 0.7)

                logger.info(
                    f"STRING PPI network for {gene}",
                    extra={
                        "gene": gene,
                        "edge_count": len(edges),
                        "avg_confidence": avg_confidence,
                        "high_confidence_edges": high_conf_count,
                        "fetch_time_ms": fetch_time
                    }
                )
            else:
                logger.warning(f"Empty PPI network for {gene} (no interactions above confidence threshold)")

            return network

        return await cached_fetch(
            source="stringdb",
            method="fetch_ppi",
            fetch_coro=_fetch_ppi,
            gene=gene,
            species=species or DEFAULT_SPECIES,
            limit=limit or MAX_INTERACTIONS
        )

    async def get_functional_annotation(self, gene: str, species: Optional[int] = None) -> Dict[str, Any]:
        """
        Get functional annotation for a gene from STRING.

        Args:
            gene: Gene symbol
            species: NCBI taxonomy ID

        Returns:
            Dict with functional annotation data
        """

        async def _fetch_annotation():
            species_id = species or DEFAULT_SPECIES

            params = {
                "identifiers": gene,
                "species": str(species_id),
                "caller_identity": "vantai.target.scoreboard"
            }

            try:
                response_data = await self._make_request("json/get_string_ids", params)

                # Parse STRING ID mapping
                if isinstance(response_data, list) and response_data:
                    string_info = response_data[0]
                    return {
                        "string_id": string_info.get("stringId", ""),
                        "preferred_name": string_info.get("preferredName", gene),
                        "protein_size": string_info.get("proteinSize", 0),
                        "annotation": string_info.get("annotation", "")
                    }
                else:
                    logger.warning(f"No STRING annotation found for {gene}")
                    return {}

            except Exception as e:
                logger.error(f"STRING annotation fetch failed for {gene}: {e}")
                return {}

        return await cached_fetch(
            source="stringdb",
            method="get_functional_annotation",
            fetch_coro=_fetch_annotation,
            gene=gene,
            species=species or DEFAULT_SPECIES
        )

    async def validate_network_quality(self, network: PPINetwork) -> DataQualityFlags:
        """
        Validate PPI network quality and set appropriate flags.

        Args:
            network: PPINetwork to validate

        Returns:
            DataQualityFlags with quality assessment
        """
        quality_flags = DataQualityFlags()

        # Check network sparsity
        if len(network.edges) == 0:
            quality_flags.partial = True
            quality_flags.notes = "Empty PPI network - no interactions found"
        elif len(network.edges) < 5:
            quality_flags.partial = True
            quality_flags.notes = f"Sparse PPI network - only {len(network.edges)} interactions"

        # Check confidence distribution
        if network.edges:
            low_conf_count = sum(1 for edge in network.edges if edge.confidence < 0.4)
            if low_conf_count > len(network.edges) * 0.8:
                quality_flags.partial = True
                quality_flags.notes = f"Low confidence network - {low_conf_count}/{len(network.edges)} edges below 0.4"

        # Check timestamp staleness (already handled by cache system)

        return quality_flags

    async def health_check(self) -> Dict[str, Any]:
        """
        Check STRING-DB API health and connectivity.

        Returns:
            Health status dict
        """
        try:
            start_time = time.time()

            # Test with a known gene
            params = {
                "identifiers": "EGFR",
                "species": str(DEFAULT_SPECIES),
                "caller_identity": "vantai.target.scoreboard"
            }

            await self._make_request("json/get_string_ids", params)

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "api_version": STRING_VERSION,
                "min_confidence": MIN_CONFIDENCE,
                "max_interactions": MAX_INTERACTIONS,
                "timestamp": get_utc_now().isoformat()
            }

        except Exception as e:
            logger.error(f"STRING health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": get_utc_now().isoformat()
            }


# ========================
# Global client management
# ========================

_stringdb_client: Optional[StringDBClient] = None


async def get_stringdb_client() -> StringDBClient:
    """Get global STRING-DB client instance."""
    global _stringdb_client
    if _stringdb_client is None:
        _stringdb_client = StringDBClient()
        logger.info("STRING-DB client initialized")
    return _stringdb_client


async def cleanup_stringdb_client() -> None:
    """Cleanup global STRING-DB client."""
    global _stringdb_client
    if _stringdb_client is not None:
        await _stringdb_client.close()
        _stringdb_client = None
        logger.info("STRING-DB client cleaned up")