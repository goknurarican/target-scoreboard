# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
PubMed eUtils API client for literature evidence.
Production-ready with polite rate limiting, metadata extraction, and quality scoring.
"""
import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import httpx

from ..schemas import EvidenceRef, DataQualityFlags, get_utc_now
from ..validation import get_validator
from .cache import cached_fetch

# Configure logger
logger = logging.getLogger(__name__)

# Configuration from environment (following NCBI eUtils guidelines)
PUBMED_BASE_URL = os.getenv("PUBMED_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "contact@vantai.com")  # Required by NCBI
PUBMED_TOOL = os.getenv("PUBMED_TOOL", "VantAI.TargetScoreboard")
REQUEST_TIMEOUT = int(os.getenv("PUBMED_TIMEOUT_SECONDS", "10"))
MAX_RETRIES = int(os.getenv("PUBMED_MAX_RETRIES", "2"))
RETRY_BACKOFF = float(os.getenv("PUBMED_RETRY_BACKOFF", "0.5"))

# Rate limiting (NCBI guidelines: max 3 requests/second, 10/second with API key)
REQUESTS_PER_SECOND = float(os.getenv("PUBMED_RATE_LIMIT", "2.0"))  # Conservative
API_KEY = os.getenv("NCBI_API_KEY")  # Optional but recommended

# Quality thresholds
HIGH_IMPACT_JOURNALS = {
    "nature", "science", "cell", "nature medicine", "nature genetics",
    "nejm", "lancet", "jama", "cancer cell", "nature cancer"
}


class PubMedError(Exception):
    """PubMed API specific errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class PubMedClient:
    """
    Production PubMed eUtils client for literature evidence.

    Features:
    - NCBI eUtils API integration with polite rate limiting
    - PMID validation and metadata extraction
    - Journal quality assessment
    - XML parsing for complete metadata
    - Cache integration with long TTL
    """

    def __init__(self, base_url: str = PUBMED_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[httpx.AsyncClient] = None
        self.validator = get_validator()
        self.rate_limiter = RateLimiter(REQUESTS_PER_SECOND)

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session with NCBI-compliant headers."""
        if self.session is None:
            headers = {
                "User-Agent": f"{PUBMED_TOOL}/1.0 (contact: {PUBMED_EMAIL})",
                "Accept": "application/xml"
            }

            self.session = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers=headers,
                limits=httpx.Limits(max_connections=3, max_keepalive_connections=2)
            )
        return self.session

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session is not None:
            await self.session.aclose()
            self.session = None

    async def _make_request(self, endpoint: str, params: Dict[str, str]) -> str:
        """
        Make rate-limited HTTP request to PubMed eUtils.

        Args:
            endpoint: eUtils endpoint (esearch, efetch, etc.)
            params: Query parameters

        Returns:
            Raw response text (usually XML)
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()

        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}.fcgi"

        # Add common parameters
        common_params = {
            "tool": PUBMED_TOOL,
            "email": PUBMED_EMAIL,
            **params
        }

        if API_KEY:
            common_params["api_key"] = API_KEY

        last_exception = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await session.get(url, params=common_params)

                # Handle errors
                if response.status_code >= 500:
                    raise PubMedError(f"PubMed server error: {response.status_code}", response.status_code)
                elif response.status_code >= 400:
                    raise PubMedError(f"PubMed client error: {response.status_code}", response.status_code)

                response.raise_for_status()
                return response.text

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"PubMed timeout, retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"PubMed error, retry {attempt + 1}/{MAX_RETRIES}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # All retries failed
        error_msg = f"PubMed API failed after {MAX_RETRIES} retries: {last_exception}"
        logger.error(error_msg, extra={"endpoint": endpoint, "params": params})
        raise PubMedError(error_msg) from last_exception

    async def fetch_pmids(self, query: str, limit: int = 20) -> List[EvidenceRef]:
        """
        Search PubMed and fetch article metadata for PMIDs.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of EvidenceRef objects with complete metadata
        """

        async def _fetch_pmids():
            if not query.strip():
                return []

            try:
                # Step 1: Search for PMIDs
                search_params = {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "xml",
                    "retmax": str(min(limit, 100)),  # Cap at 100 for performance
                    "sort": "relevance"
                }

                search_response = await self._make_request("esearch", search_params)
                pmids = self._parse_pmids_from_search(search_response)

                if not pmids:
                    logger.info(f"No PMIDs found for query: {query}")
                    return []

                # Step 2: Fetch article metadata for PMIDs
                evidence_refs = []

                # Process PMIDs in batches to respect rate limits
                batch_size = 10
                for i in range(0, len(pmids), batch_size):
                    batch_pmids = pmids[i:i + batch_size]

                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(batch_pmids),
                        "retmode": "xml",
                        "rettype": "abstract"
                    }

                    fetch_response = await self._make_request("efetch", fetch_params)
                    batch_evidence = self._parse_article_metadata(fetch_response)
                    evidence_refs.extend(batch_evidence)

                logger.info(
                    f"PubMed search completed",
                    extra={
                        "query": query[:100],
                        "pmids_found": len(pmids),
                        "metadata_fetched": len(evidence_refs)
                    }
                )

                return evidence_refs

            except Exception as e:
                logger.error(f"PubMed search failed for query '{query}': {e}")
                return []

        return await cached_fetch(
            source="pubmed",
            method="fetch_pmids",
            fetch_coro=_fetch_pmids,
            query=query,
            limit=limit
        )

    def _parse_pmids_from_search(self, xml_response: str) -> List[str]:
        """Parse PMIDs from esearch XML response."""
        try:
            root = ET.fromstring(xml_response)
            pmids = []

            for id_elem in root.findall(".//IdList/Id"):
                pmid = id_elem.text
                if pmid:
                    pmids.append(pmid)

            return pmids

        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed search response: {e}")
            return []

    def _parse_article_metadata(self, xml_response: str) -> List[EvidenceRef]:
        """Parse article metadata from efetch XML response."""
        evidence_refs = []

        try:
            root = ET.fromstring(xml_response)

            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text

                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else None

                    # Extract journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else None

                    # Extract year
                    year = None
                    year_elem = article.find(".//PubDate/Year")
                    if year_elem is not None:
                        try:
                            year = int(year_elem.text)
                        except ValueError:
                            pass

                    # Assess source quality
                    source_quality = self._assess_journal_quality(journal)

                    # Create evidence reference
                    evidence = EvidenceRef(
                        source="pubmed",
                        pmid=pmid,
                        title=title,
                        journal=journal,
                        year=year,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        source_quality=source_quality,
                        timestamp=get_utc_now()
                    )

                    evidence_refs.append(evidence)

                except Exception as e:
                    logger.warning(f"Error parsing article metadata: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed fetch response: {e}")

        return evidence_refs

    def _assess_journal_quality(self, journal_name: Optional[str]) -> str:
        """
        Assess journal quality based on name matching.

        Args:
            journal_name: Journal name string

        Returns:
            Quality level string
        """
        if not journal_name:
            return "unknown"

        journal_lower = journal_name.lower()

        # Check high-impact journals
        for high_impact in HIGH_IMPACT_JOURNALS:
            if high_impact in journal_lower:
                return "high"

        # Check for review journals
        if any(term in journal_lower for term in ["review", "reviews"]):
            return "medium"

        # Default quality
        return "medium"

    async def validate_pmid(self, pmid: str) -> Dict[str, Any]:
        """
        Validate PMID and get basic metadata.

        Args:
            pmid: PubMed ID to validate

        Returns:
            Dict with validation status and metadata
        """

        async def _validate_pmid():
            try:
                params = {
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml",
                    "rettype": "abstract"
                }

                response = await self._make_request("efetch", params)
                evidence_refs = self._parse_article_metadata(response)

                if evidence_refs:
                    evidence = evidence_refs[0]
                    return {
                        "valid": True,
                        "title": evidence.title,
                        "journal": evidence.journal,
                        "year": evidence.year,
                        "quality": evidence.source_quality
                    }
                else:
                    return {"valid": False, "error": "PMID not found or invalid"}

            except Exception as e:
                logger.error(f"PMID validation failed for {pmid}: {e}")
                return {"valid": False, "error": str(e)}

        return await cached_fetch(
            source="pubmed",
            method="validate_pmid",
            fetch_coro=_validate_pmid,
            pmid=pmid
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check PubMed eUtils API health.

        Returns:
            Health status dict
        """
        try:
            start_time = time.time()

            # Test search with simple query
            params = {
                "db": "pubmed",
                "term": "cancer",
                "retmode": "xml",
                "retmax": "1"
            }

            await self._make_request("esearch", params)

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "rate_limit": f"{REQUESTS_PER_SECOND} req/sec",
                "api_key_configured": bool(API_KEY),
                "timestamp": get_utc_now().isoformat()
            }

        except Exception as e:
            logger.error(f"PubMed health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": get_utc_now().isoformat()
            }


class RateLimiter:
    """Simple async rate limiter for API requests."""

    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit slot (blocks if necessary)."""
        async with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time

            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                await asyncio.sleep(sleep_time)

            self.last_request_time = time.time()


# ========================
# Global client management
# ========================

_pubmed_client: Optional[PubMedClient] = None


async def get_pubmed_client() -> PubMedClient:
    """Get global PubMed client instance."""
    global _pubmed_client
    if _pubmed_client is None:
        _pubmed_client = PubMedClient()
        logger.info("PubMed client initialized")
    return _pubmed_client


async def cleanup_pubmed_client() -> None:
    """Cleanup global PubMed client."""
    global _pubmed_client
    if _pubmed_client is not None:
        await _pubmed_client.close()
        _pubmed_client = None
        logger.info("PubMed client cleaned up")


# ========================
# Convenience functions for common searches
# ========================

async def search_gene_disease_literature(gene: str, disease: str, limit: int = 10) -> List[EvidenceRef]:
    """
    Search for literature linking gene and disease.

    Args:
        gene: Gene symbol
        disease: Disease term
        limit: Maximum results

    Returns:
        List of EvidenceRef objects
    """
    client = await get_pubmed_client()

    # Build focused search query
    query = f'("{gene}"[Gene Name] OR "{gene}"[Title/Abstract]) AND ("{disease}"[Title/Abstract] OR "{disease}"[MeSH Terms])'

    return await client.fetch_pmids(query, limit)


async def validate_pmid_list(pmids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Validate a list of PMIDs and return metadata.

    Args:
        pmids: List of PMID strings

    Returns:
        Dict mapping PMID to validation result
    """
    client = await get_pubmed_client()

    results = {}
    for pmid in pmids:
        try:
            validation_result = await client.validate_pmid(pmid)
            results[pmid] = validation_result
        except Exception as e:
            logger.error(f"PMID validation failed for {pmid}: {e}")
            results[pmid] = {"valid": False, "error": str(e)}

    return results