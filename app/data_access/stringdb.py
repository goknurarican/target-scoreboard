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
from typing import Dict, List, Optional, Any, Set
from urllib.parse import quote
import json as _json

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
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"

        # DEBUG: Request details
        logger.info(f"STRING API Request: {url}")
        logger.info(f"STRING API Params: {params}")

        last_exception = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                # headers override KALDIRILDI
                response = await session.get(url, params=params)

                # DEBUG: Response details
                logger.info(f"STRING API Status: {response.status_code}")
                logger.info(f"STRING API Headers: {dict(response.headers)}")
                logger.info(f"STRING API Response length: {len(response.content)}")

                if response.status_code == 429:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"STRING API rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    raise StringDBError(f"STRING server error: {response.status_code}", response.status_code)
                elif response.status_code >= 400:
                    body_preview = response.text[:200].replace("\n", " ")
                    raise StringDBError(f"STRING client error: {response.status_code} body={body_preview}",
                                        response.status_code)

                # ---- İçerik ayrıştırma: endpoint'e göre ----
                if endpoint.startswith("json/"):
                    try:
                        result = response.json()
                        logger.info(f"STRING JSON parsed successfully: {len(str(result))} chars")
                        return result
                    except Exception:
                        try:
                            result = _json.loads(response.text)
                            logger.info(f"STRING JSON manual parse success: {len(str(result))} chars")
                            return result
                        except Exception as je:
                            logger.error(f"STRING JSON parse failed: {je}")
                            logger.error(f"Response text: {response.text[:500]}")
                            raise StringDBError(f"Invalid JSON from {endpoint}: {response.text[:200]}",
                                                response.status_code) from je

                if endpoint.startswith("tsv/"):
                    result = {"text_data": response.text}
                    logger.info(f"STRING TSV response: {len(response.text)} chars")
                    logger.info(f"TSV first 500 chars: {response.text[:500]}")
                    return result

                # güvenli varsayılanlar
                ct = (response.headers.get("content-type") or "").lower()
                if "json" in ct:
                    return response.json()
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
                logger.error(f"STRING request error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES and isinstance(e, (httpx.TransportError, StringDBError)):
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
        Strategy: map gene -> stringId via get_string_ids, then call TSV endpoint.
        """
        logger.info(f"fetch_ppi called for gene={gene}, species={species}, limit={limit}")

        async def _fetch_ppi():
            logger.info(f"_fetch_ppi inner function called for {gene}")

            species_id = species or DEFAULT_SPECIES
            interaction_limit = min(limit or MAX_INTERACTIONS, MAX_INTERACTIONS)

            # 1) Gene symbol -> STRING ID
            map_params = {
                "identifiers": gene,
                "species": str(species_id),
                "limit": "1",
                "caller_identity": "vantai.target.scoreboard"
            }
            map_resp = await self._make_request("json/get_string_ids", map_params)
            if not isinstance(map_resp, list) or not map_resp or "stringId" not in map_resp[0]:
                logger.warning(f"STRING mapping failed for {gene} (no stringId); raw={str(map_resp)[:180]}")
                return PPINetwork(
                    gene=gene.upper(),
                    edges=[],
                    timestamp=get_utc_now(),
                    source=f"stringdb_v{STRING_VERSION}"
                )
            string_id = map_resp[0]["stringId"]
            preferred_name = map_resp[0].get("preferredName", gene)

            # ---------------------------
            # 2a) Hedefin partnerları (1-hop)
            # ---------------------------
            params_partners = {
                "identifiers": string_id,
                "species": str(species_id),
                "required_score": str(int(MIN_CONFIDENCE * 1000)),  # 0-1000
                "network_type": "functional",
                "limit": str(interaction_limit),
                "caller_identity": "vantai.target.scoreboard"
            }
            t0 = time.time()
            resp_partners = await self._make_request("tsv/interaction_partners", params_partners)
            fetch_time_partners = (time.time() - t0) * 1000

            edges: List[PPIEdge] = []
            partner_ids: List[str] = []

            # TSV parse (partner listesi + hedef-partner kenarları)
            tsv_text = resp_partners.get("text_data") if isinstance(resp_partners, dict) else str(resp_partners)
            if tsv_text and tsv_text.strip():
                lines = [ln for ln in tsv_text.strip().splitlines() if ln.strip()]
                # header satırını bul
                header_idx = None
                for i, ln in enumerate(lines):
                    if ln.startswith("#"):
                        continue
                    header_idx = i
                    break
                if header_idx is None:
                    logger.warning(f"No TSV header for {preferred_name}")
                else:
                    headers = lines[header_idx].split("\t")
                    col = {h: idx for idx, h in enumerate(headers)}
                    # beklenen sütun isimleri: stringId_A, stringId_B, preferredName_A, preferredName_B, ncbiTaxonId, score
                    for i, line in enumerate(lines[header_idx + 1:], 1):
                        if not line.strip():
                            continue
                        cols = line.split("\t")
                        if len(cols) < 6:
                            continue
                        sid_b = cols[col.get("stringId_B", 1)].strip()
                        pref_a = cols[col.get("preferredName_A", 2)].strip()
                        pref_b = cols[col.get("preferredName_B", 3)].strip()
                        score_str = cols[col.get("score", 5)].strip()
                        try:
                            conf = float(score_str)
                        except:
                            continue
                        if conf < MIN_CONFIDENCE:
                            continue

                        if sid_b:
                            partner_ids.append(sid_b)

                        edges.append(PPIEdge(
                            source_gene=pref_a.upper(),
                            partner=pref_b.upper(),
                            confidence=conf,
                            evidence=[EvidenceRef(
                                source="stringdb",
                                title=f"STRING interaction: {pref_a} - {pref_b}",
                                url=f"https://string-db.org/network/{string_id}",
                                source_quality="high" if conf >= 0.7 else "medium",
                                timestamp=get_utc_now()
                            )]
                        ))
                logger.info(f"Parsed {len(edges)} target-partner edges for {preferred_name}")
            else:
                logger.warning(f"Empty partners TSV for {preferred_name}")

            # -----------------------------------------------
            # 2b) Alt-ağ: hedef + partnerların kendi araları
            # -----------------------------------------------
            fetch_time_network = 0.0
            if partner_ids:
                # STRING çoklu ID’yi newline ile kabul eder
                ids_blob = "\n".join([string_id] + partner_ids[:interaction_limit])
                params_network = {
                    "identifiers": ids_blob,
                    "species": str(species_id),
                    "required_score": str(int(MIN_CONFIDENCE * 1000)),
                    "caller_identity": "vantai.target.scoreboard"
                }
                t1 = time.time()
                resp_network = await self._make_request("tsv/network", params_network)
                fetch_time_network = (time.time() - t1) * 1000

                net_text = resp_network.get("text_data") if isinstance(resp_network, dict) else str(resp_network)
                if net_text and net_text.strip():
                    lines = [ln for ln in net_text.strip().splitlines() if ln.strip()]
                    # header satırını bul
                    header_idx = None
                    for i, ln in enumerate(lines):
                        if ln.startswith("#"):
                            continue
                        header_idx = i
                        break
                    if header_idx is not None:
                        headers = lines[header_idx].split("\t")
                        col = {h: idx for idx, h in enumerate(headers)}
                        seen: Set[tuple] = set()
                        for i, line in enumerate(lines[header_idx + 1:], 1):
                            cols = line.split("\t")
                            if len(cols) < 6:
                                continue
                            pref_a = cols[col.get("preferredName_A", 2)].strip().upper()
                            pref_b = cols[col.get("preferredName_B", 3)].strip().upper()
                            score_str = cols[col.get("score", 5)].strip()
                            try:
                                conf = float(score_str)
                            except:
                                continue
                            if conf < MIN_CONFIDENCE:
                                continue

                            a, b = (pref_a, pref_b) if pref_a <= pref_b else (pref_b, pref_a)
                            if (a, b) in seen:
                                continue
                            seen.add((a, b))

                            edges.append(PPIEdge(
                                source_gene=a,
                                partner=b,
                                confidence=conf,
                                evidence=[EvidenceRef(
                                    source="stringdb",
                                    title=f"STRING interaction: {a} - {b}",
                                    url=f"https://string-db.org/network/{string_id}",
                                    source_quality="high" if conf >= 0.7 else "medium",
                                    timestamp=get_utc_now()
                                )]
                            ))
                    else:
                        logger.warning(f"No subgraph TSV header for {preferred_name}")
                else:
                    logger.warning(f"No subgraph TSV for {preferred_name}")
            else:
                logger.warning(f"No partners for {preferred_name}; subgraph skipped")

            fetch_time = fetch_time_partners + fetch_time_network

            # 3) Network objesi
            network = PPINetwork(
                gene=preferred_name.upper(),
                edges=edges,
                timestamp=get_utc_now(),
                source=f"stringdb_v{STRING_VERSION}"
            )

            # 4) Log / özet
            if edges:
                avg_conf = sum(e.confidence for e in edges) / len(edges)
                hi = sum(1 for e in edges if e.confidence >= 0.7)
                logger.info(
                    f"STRING PPI network for {preferred_name}: {len(edges)} edges, avg_conf={avg_conf:.3f}, high_conf={hi}",
                    extra={
                        "gene": preferred_name,
                        "edge_count": len(edges),
                        "avg_confidence": avg_conf,
                        "high_confidence_edges": hi,
                        "fetch_time_ms": fetch_time
                    }
                )
            else:
                logger.warning(f"Empty PPI network for {preferred_name} (no interactions above threshold {MIN_CONFIDENCE})")

            return network

        return await _fetch_ppi()

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
                "limit": "1",
                "caller_identity": "vantai.target.scoreboard"
            }
            try:
                resp = await self._make_request("json/get_string_ids", params)
                if isinstance(resp, list) and resp:
                    info = resp[0]
                    return {
                        "string_id": info.get("stringId", ""),
                        "preferred_name": info.get("preferredName", gene),
                        "protein_size": info.get("proteinSize", 0),
                        "annotation": info.get("annotation", "")
                    }
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