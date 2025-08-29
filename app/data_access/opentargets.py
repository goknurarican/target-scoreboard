# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
OpenTargets GraphQL API client - Production ready with real data integration.
No demo modes, no synthetic fallbacks. Real biological data only.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import httpx

from ..schemas import (
    AssociationRecord,
    EvidenceRef,
    ValidationResult,
    DataQualityFlags,
    get_utc_now
)
from ..validation import get_validator
from .cache import cached_fetch

# Configure logger
logger = logging.getLogger(__name__)

# Configuration
OT_GRAPHQL_URL = os.getenv("OT_GRAPHQL_URL", "https://api.platform.opentargets.org/api/v4/graphql")
OT_REST_URL = os.getenv("OT_REST_URL", "https://api.platform.opentargets.org/api/v4")
MAX_RETRIES = int(os.getenv("OT_MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("OT_RETRY_BACKOFF", "1.0"))
REQUEST_TIMEOUT = int(os.getenv("OT_TIMEOUT_SECONDS", "30"))


class OpenTargetsError(Exception):
    """OpenTargets API specific errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class OpenTargetsClient:
    """
    Production OpenTargets API client.
    
    Features:
    - Real GraphQL/REST API integration
    - Async + retry with exponential backoff  
    - Schema validation using DataQualityValidator
    - Production cache integration
    - Structured error logging
    """

    def __init__(self, base_url: str = OT_GRAPHQL_URL):
        self.base_url = base_url
        self.rest_base = OT_REST_URL
        self.session: Optional[httpx.AsyncClient] = None
        self.validator = get_validator()

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self.session is None:
            # http2'yi 'h2' yoksa kapat
            try:
                import h2  # noqa: F401
                use_http2 = bool(int(os.getenv("OT_HTTP2", "1")))
            except Exception:
                use_http2 = False
            self.session = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "VantAI-TargetScoreboard/1.0"
                },
                http2=use_http2,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self.session

    async def close(self) -> None:
        """Close HTTP session."""
        if self.session is not None:
            await self.session.aclose()
            self.session = None

    async def _execute_graphql(self, query: str, variables: Dict) -> Tuple[Dict, float]:
        """
        Execute GraphQL query with retries and error handling.
        
        Returns:
            (response_data, fetch_time_ms)
        """
        start_time = time.time()
        session = await self._get_session()
        payload = {"query": query, "variables": variables}

        last_exception = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await session.post(self.base_url, json=payload)
                
                # Handle HTTP errors
                if response.status_code == 429:
                    wait_time = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(f"Rate limited by OpenTargets, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                elif response.status_code >= 500:
                    raise OpenTargetsError(f"Server error: {response.status_code}", response.status_code)
                elif response.status_code >= 400:
                    raise OpenTargetsError(f"Client error: {response.status_code}", response.status_code)

                response.raise_for_status()
                data = response.json()

                # Handle GraphQL errors
                if "errors" in data:
                    error_msg = "; ".join([e.get("message", str(e)) for e in data["errors"]])
                    raise OpenTargetsError(f"GraphQL errors: {error_msg}", response_data=data)

                fetch_time = (time.time() - start_time) * 1000
                return data, fetch_time

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(f"OpenTargets timeout, retry {attempt+1}/{MAX_RETRIES} in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES and isinstance(e, (httpx.NetworkError, OpenTargetsError)):
                    wait_time = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(f"OpenTargets error, retry {attempt+1}/{MAX_RETRIES}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        # All retries failed
        error_msg = f"OpenTargets API failed after {MAX_RETRIES} retries: {last_exception}"
        logger.error(error_msg, extra={"query": query, "variables": variables})
        raise OpenTargetsError(error_msg) from last_exception

    async def get_target_id(self, gene_symbol: str) -> Optional[str]:
        """
        Resolve gene symbol to ENSG ID via OpenTargets search.
        
        Args:
            gene_symbol: Gene symbol (e.g., 'EGFR')
            
        Returns:
            ENSG ID or None if not found
        """
        # If already ENSG, return as-is
        gene_symbol = gene_symbol.strip().upper()
        if gene_symbol.startswith("ENSG"):
            return gene_symbol

        async def _fetch_target_id():
            query = """
            query FindTarget($q: String!) {
                search(queryString: $q, entityNames: ["target"]) {
                    hits {
                        id
                        name
                        entity
                        object {
                            ... on Target {
                                approvedSymbol
                                alternativeSymbols
                            }
                        }
                    }
                }
            }
            """
            
            variables = {"q": gene_symbol}
            data, fetch_time = await self._execute_graphql(query, variables)
            
            # Parse results
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            
            # Look for exact symbol match first
            for hit in hits:
                if hit.get("entity") == "target":
                    target_obj = hit.get("object", {})
                    approved = target_obj.get("approvedSymbol", "").upper()
                    alternatives = target_obj.get("alternativeSymbols", [])
                    
                    if approved == gene_symbol or gene_symbol in [alt.upper() for alt in alternatives]:
                        ensg_id = hit.get("id", "")
                        if ensg_id.startswith("ENSG"):
                            logger.info(f"Gene symbol resolved: {gene_symbol} -> {ensg_id}")
                            return ensg_id

            # No exact match found
            logger.warning(f"Gene symbol not found in OpenTargets: {gene_symbol}")
            return None

        try:
            return await cached_fetch(
                source="opentargets",
                method="get_target_id",
                fetch_coro=_fetch_target_id,
                gene_symbol=gene_symbol
            )
        except Exception as e:
            logger.error(f"Failed to resolve gene symbol {gene_symbol}: {e}")
            return None

    async def fetch_gene_disease_associations(self, gene: str, disease: str) -> List[AssociationRecord]:
        """
        Fetch gene-disease associations from OpenTargets.
        
        Args:
            gene: Gene symbol or ENSG ID
            disease: Disease EFO ID
            
        Returns:
            List of AssociationRecord objects
        """
        async def _fetch_associations():
            # Resolve gene symbol to ENSG
            ensg_id = await self.get_target_id(gene)
            if not ensg_id:
                logger.warning(f"Cannot resolve gene {gene} to ENSG ID")
                return []

            query = """
            query GetAssociations($diseaseId: String!, $targetId: String!) {
                associationDiseaseTarget(diseaseId: $diseaseId, targetId: $targetId) {
                    id
                    score
                    datatypeScores {
                        id
                        score
                    }
                }
                evidences(
                    diseaseId: $diseaseId
                    targetId: $targetId
                    size: 100
                ) {
                    count
                    rows {
                        disease {
                            id
                            name
                        }
                        target {
                            id
                            approvedSymbol
                        }
                        score
                        scoreExponent
                        datasourceId
                        datatypeId
                        literature
                    }
                }
                meta {
                    apiVersion {
                        major
                        minor
                        patch
                    }
                    dataVersion {
                        year
                        month
                        iteration
                    }
                }
            }
            """

            variables = {"diseaseId": disease, "targetId": ensg_id}
            data, fetch_time = await self._execute_graphql(query, variables)

            return self._parse_associations(data, gene, disease, fetch_time)

        return await cached_fetch(
            source="opentargets",
            method="fetch_gene_disease_associations",
            fetch_coro=_fetch_associations,
            gene=gene,
            disease=disease
        )

    async def fetch_evidences(self, gene: str, disease: str) -> List[EvidenceRef]:
        """
        Fetch evidence references for gene-disease pair.
        
        Args:
            gene: Gene symbol
            disease: Disease EFO ID
            
        Returns:
            List of EvidenceRef objects
        """
        async def _fetch_evidences():
            # Get associations first to extract evidence
            associations = await self.fetch_gene_disease_associations(gene, disease)
            
            evidence_refs = []
            for assoc in associations:
                evidence_refs.extend(assoc.evidence)

            # Deduplicate by PMID if available
            seen_pmids = set()
            unique_evidence = []
            
            for evidence in evidence_refs:
                if evidence.pmid:
                    if evidence.pmid not in seen_pmids:
                        seen_pmids.add(evidence.pmid)
                        unique_evidence.append(evidence)
                else:
                    unique_evidence.append(evidence)

            return unique_evidence

        return await cached_fetch(
            source="opentargets",
            method="fetch_evidences",
            fetch_coro=_fetch_evidences,
            gene=gene,
            disease=disease
        )

    def _parse_associations(self, api_response: Dict, gene: str, disease: str, fetch_time: float) -> List[AssociationRecord]:
        """Parse OpenTargets API response into AssociationRecord objects."""
        associations = []
        
        try:
            # Extract main association data
            adt = api_response.get("data", {}).get("associationDiseaseTarget", {})
            evidences_data = api_response.get("data", {}).get("evidences", {})
            meta = api_response.get("data", {}).get("meta", {})

            # Parse evidence references
            evidence_refs = []
            for evidence_row in evidences_data.get("rows", []):
                literature = evidence_row.get("literature", [])
                if literature:
                    for lit in literature:
                        evidence_refs.append(EvidenceRef(
                            source="opentargets",
                            pmid=str(lit) if lit else None,
                            title=None,  # Will be filled by PubMed client
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{lit}/" if lit else None,
                            source_quality="high",
                            timestamp=get_utc_now()
                        ))

            # Create main association record
            if adt:
                overall_score = float(adt.get("score", 0.0))
                
                # Extract genetics score from datatypes
                genetics_score = 0.0
                for datatype in adt.get("datatypeScores", []):
                    datatype_id = datatype.get("id", "").lower()
                    if datatype_id in ("genetic_association", "genetics"):
                        genetics_score = max(genetics_score, float(datatype.get("score", 0.0)))

                association = AssociationRecord(
                    gene=gene.upper(),
                    disease=disease,
                    score=overall_score,
                    pval=None,  # Not available in this query
                    source="opentargets",
                    timestamp=get_utc_now(),
                    evidence=evidence_refs
                )

                associations.append(association)

                # Add genetics-specific record if different
                if genetics_score > 0 and genetics_score != overall_score:
                    genetics_association = AssociationRecord(
                        gene=gene.upper(),
                        disease=disease,
                        score=genetics_score,
                        pval=None,
                        source="opentargets_genetics",
                        timestamp=get_utc_now(),
                        evidence=evidence_refs
                    )
                    associations.append(genetics_association)

            # Validate parsed data
            for association in associations:
                validation_result = self.validator.validate("opentargets", association)
                if not validation_result.ok:
                    logger.warning(
                        f"Association validation failed: {validation_result.issues}",
                        extra={
                            "gene": gene,
                            "disease": disease,
                            "issues": validation_result.issues
                        }
                    )

            logger.info(
                f"Parsed {len(associations)} associations for {gene}-{disease}",
                extra={
                    "gene": gene,
                    "disease": disease,
                    "association_count": len(associations),
                    "evidence_count": len(evidence_refs),
                    "fetch_time_ms": fetch_time
                }
            )

        except Exception as e:
            logger.error(f"Error parsing OpenTargets response: {e}", extra={"gene": gene, "disease": disease})
            # Return empty list instead of raising - let caller handle missing data
            return []

        return associations

    async def fetch_tractability(self, gene: str) -> Optional[Dict[str, Any]]:
        """
        Fetch tractability information for a gene.
        
        Args:
            gene: Gene symbol
            
        Returns:
            Tractability data dict or None if not available
        """
        async def _fetch_tractability():
            ensg_id = await self.get_target_id(gene)
            if not ensg_id:
                return None

            query = """
            query GetTractability($targetId: String!) {
                target(ensemblId: $targetId) {
                    id
                    approvedSymbol
                    tractability {
                        id
                        modality
                        value
                    }
                }
            }
            """

            variables = {"targetId": ensg_id}
            
            try:
                data, fetch_time = await self._execute_graphql(query, variables)
                
                target_data = data.get("data", {}).get("target", {})
                tractability = target_data.get("tractability", [])
                
                # Parse tractability into structured dict
                tractability_dict = {}
                for tract in tractability:
                    modality = tract.get("modality", "").lower()
                    value = tract.get("value")
                    if modality and value is not None:
                        tractability_dict[modality] = float(value)

                logger.info(
                    f"Fetched tractability for {gene}",
                    extra={
                        "gene": gene,
                        "ensg_id": ensg_id,
                        "modalities": list(tractability_dict.keys()),
                        "fetch_time_ms": fetch_time
                    }
                )

                return tractability_dict if tractability_dict else None

            except Exception as e:
                logger.warning(f"Tractability fetch failed for {gene}: {e}")
                return None

        return await cached_fetch(
            source="opentargets",
            method="fetch_tractability",
            fetch_coro=_fetch_tractability,
            gene=gene
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check OpenTargets API health and connectivity.
        
        Returns:
            Health status dict
        """
        try:
            start_time = time.time()
            
            # Simple ping query
            query = """
            query HealthCheck {
                meta {
                    apiVersion {
                        major
                        minor
                    }
                    dataVersion {
                        year
                        month
                    }
                }
            }
            """
            
            data, _ = await self._execute_graphql(query, {})
            
            response_time = (time.time() - start_time) * 1000
            meta = data.get("data", {}).get("meta", {})
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "api_version": meta.get("apiVersion", {}),
                "data_version": meta.get("dataVersion", {}),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"OpenTargets health check failed: {e}")
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# ========================
# Global client management
# ========================

_ot_client: Optional[OpenTargetsClient] = None


async def get_ot_client() -> OpenTargetsClient:
    """Get global OpenTargets client instance."""
    global _ot_client
    if _ot_client is None:
        _ot_client = OpenTargetsClient()
        logger.info("OpenTargets client initialized")
    return _ot_client


async def cleanup_ot_client() -> None:
    """Cleanup global OpenTargets client."""
    global _ot_client
    if _ot_client is not None:
        await _ot_client.close()
        _ot_client = None
        logger.info("OpenTargets client cleaned up")


# ========================
# High-level convenience functions
# ========================

async def fetch_ot_association(disease_efo: str, target_symbol_or_ensg: str) -> Dict:
    """
    High-level convenience function for gene-disease associations.
    
    Args:
        disease_efo: Disease EFO ID
        target_symbol_or_ensg: Gene symbol or ENSG ID
        
    Returns:
        Compact dict with association scores and metadata
    """
    try:
        client = await get_ot_client()
        associations = await client.fetch_gene_disease_associations(target_symbol_or_ensg, disease_efo)
        
        if not associations:
            return {
                "overall": 0.0,
                "genetics": 0.0,
                "text_mining": 0.0,
                "known_drug": 0.0,
                "evidence_count": 0,
                "release": "OT-unknown",
                "cached": False,
                "fetch_ms": 0.0,
                "status": "data_missing"
            }

        # Extract scores from associations
        overall_score = 0.0
        genetics_score = 0.0
        evidence_count = 0

        for assoc in associations:
            if assoc.source == "opentargets":
                overall_score = max(overall_score, assoc.score)
            elif assoc.source == "opentargets_genetics":
                genetics_score = max(genetics_score, assoc.score)
            evidence_count += len(assoc.evidence)

        return {
            "overall": overall_score,
            "genetics": genetics_score,
            "text_mining": 0.0,  # TODO: Extract from evidence types
            "known_drug": 0.0,   # TODO: Extract from evidence types  
            "evidence_count": evidence_count,
            "release": "OT-24.12",  # TODO: Extract from meta
            "cached": True,  # TODO: Track cache hit from fetch
            "fetch_ms": 0.0,  # TODO: Track actual fetch time
            "status": "ok"
        }

    except Exception as e:
        logger.error(f"Association fetch failed: {e}", extra={"gene": target_symbol_or_ensg, "disease": disease_efo})
        return {
            "overall": 0.0,
            "genetics": 0.0,
            "text_mining": 0.0,
            "known_drug": 0.0,
            "evidence_count": 0,
            "release": "OT-unknown",
            "cached": False,
            "fetch_ms": 0.0,
            "status": "error",
            "error": str(e)
        }