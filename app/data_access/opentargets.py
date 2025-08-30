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

NETWORK_ERRORS = (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)

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
                if response.status_code >= 400:
                    # Daha açıklayıcı hata
                    try:
                        j = response.json()
                        msg = "; ".join([e.get("message", str(e)) for e in j.get("errors", [])]) or response.text
                    except Exception:
                        msg = response.text
                    raise OpenTargetsError(f"Client error: {response.status_code} - {msg}", response.status_code)
                response.raise_for_status()
                data = response.json()
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
                if attempt < MAX_RETRIES and isinstance(e, (OpenTargetsError,) + NETWORK_ERRORS):
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
        gene_symbol = gene_symbol.strip().upper()
        if gene_symbol.startswith("ENSG"):
            return gene_symbol

        async def _fetch_target_id():
            query = """
            query TargetSearch($q:String!){
              search(queryString:$q, entityNames:["target"]){
                hits{
                  id
                  entity
                  name
                  object{
                    ... on Target{
                      approvedSymbol
                      synonyms{ label }
                      symbolSynonyms{ label }
                    }
                  }
                }
              }
            }
            """
            variables = {"q": gene_symbol}
            data, _ = await self._execute_graphql(query, variables)
            hits = (((data or {}).get("data") or {}).get("search") or {}).get("hits", [])
            for hit in hits:
                if (hit.get("entity") or "").lower() != "target":
                    continue
                obj = hit.get("object") or {}
                approved = (obj.get("approvedSymbol") or "").upper()
                syns = [(s.get("label") or "").upper() for s in (obj.get("synonyms") or [])]
                syns2 = [(s.get("label") or "").upper() for s in (obj.get("symbolSynonyms") or [])]
                if gene_symbol == approved or gene_symbol in syns or gene_symbol in syns2:
                    ensg_id = hit.get("id") or ""
                    if ensg_id.startswith("ENSG"):
                        logger.info(f"Gene symbol resolved: {gene_symbol} -> {ensg_id}")
                        return ensg_id
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
        async def _fetch_associations():
            ensg_id = await self.get_target_id(gene)
            if not ensg_id:
                logger.warning(f"Cannot resolve gene {gene} to ENSG ID")
                return []

            query = """
            query GetAssoc($targetId:String!,$diseaseId:String!,$page:Pagination){
              target(ensemblId:$targetId){
                associatedDiseases(
                  Bs:[$diseaseId],
                  enableIndirect:true,
                  page:$page
                ){
                  count
                  rows{
                    score
                    datatypeScores{ id score }
                    disease{ id name }
                  }
                }
              }
              disease(efoId:$diseaseId){
                associatedTargets(
                  Bs:[$targetId],
                  enableIndirect:true,
                  page:$page
                ){
                  count
                  rows{
                    score
                    datatypeScores{ id score }
                    target{ id approvedSymbol }
                  }
                }
                evidences(
                  ensemblIds:[$targetId],
                  enableIndirect:true,
                  datasourceIds:["ot_genetics_portal"],
                  size:1
                ){ count }
              }
              meta{
                apiVersion{ x y z }
                dataVersion{ year month iteration }
              }
            }
            """
            variables = {"targetId": ensg_id, "diseaseId": disease, "page": {"index": 0, "size": 1}}
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

    def _parse_associations(self, api_response: Dict, gene: str, disease: str, fetch_time: float) -> List[
        AssociationRecord]:
        associations: List[AssociationRecord] = []
        try:
            tgt = (api_response.get("data", {}) or {}).get("target", {}) or {}
            dis = (api_response.get("data", {}) or {}).get("disease", {}) or {}

            # Target -> associatedDiseases
            t_rows = ((tgt.get("associatedDiseases") or {}).get("rows") or [])
            # Disease -> associatedTargets
            d_rows = ((dis.get("associatedTargets") or {}).get("rows") or [])

            rows = []
            if t_rows: rows.append(t_rows[0])
            if d_rows: rows.append(d_rows[0])

            # Evidence sayısı
            evidence_count = (((dis.get("evidences") or {}).get("count")) or 0)

            overall = 0.0
            genetics = 0.0

            for r in rows:
                overall = max(overall, float(r.get("score") or 0.0))
                for ds in (r.get("datatypeScores") or []):
                    did = str(ds.get("id") or "").lower()
                    if ("genetic" in did) or ("somatic" in did):
                        genetics = max(genetics, float(ds.get("score") or 0.0))

            # Her iki yön de boşsa boş liste döndür (çağıran fallback’ı halleder)
            if overall == 0.0 and genetics == 0.0 and evidence_count == 0:
                return []

            # Tek bir birleşik kayıt üret
            associations.append(
                AssociationRecord(
                    gene=gene.upper(),
                    disease=disease,
                    score=overall if overall > 0 else genetics,
                    pval=None,
                    source="opentargets",
                    timestamp=get_utc_now(),
                    evidence=[EvidenceRef(source="opentargets", title=f"OT evidences: {evidence_count}",
                                          timestamp=get_utc_now())]
                )
            )

            # (İsteğe bağlı) genetics ayrı bir kayıt olarak farklıysa ekle
            if genetics > 0 and genetics != overall:
                associations.append(
                    AssociationRecord(
                        gene=gene.upper(),
                        disease=disease,
                        score=genetics,
                        pval=None,
                        source="opentargets_genetics",
                        timestamp=get_utc_now(),
                        evidence=[]
                    )
                )

            # doğrulama
            for a in associations:
                vr = self.validator.validate("opentargets", a)
                if not vr.ok:
                    logger.warning("Association validation issues: %s", vr.issues)

            logger.info(
                "Parsed associations for %s-%s overall=%.3f genetics=%.3f evidences=%d (%.1fms)",
                gene, disease, overall, genetics, evidence_count, fetch_time
            )
        except Exception as e:
            logger.error("Error parsing OpenTargets response: %s", e, extra={"gene": gene, "disease": disease})
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
        """
        try:
            start_time = time.time()
            query = """
            query HealthCheck {
              meta {
                name
                apiVersion { x y z }
                dataVersion { year month iteration }
              }
            }
            """
            data, _ = await self._execute_graphql(query, {})
            response_time = (time.time() - start_time) * 1000

            meta = (data.get("data") or {}).get("meta") or {}
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "api_version": meta.get("apiVersion", {}),
                "data_version": meta.get("dataVersion", {}),
                "name": meta.get("name"),
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
        logger.info("OpenTargets client initialized (mode=live, url=%s)", OT_GRAPHQL_URL)
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
    try:
        client = await get_ot_client()
        associations = await client.fetch_gene_disease_associations(target_symbol_or_ensg, disease_efo)

        if not associations:
            return {
                "overall": 0.0, "overallScore": 0.0, "genetics": 0.0,
                "evidence_count": 0, "release": "OT-unknown",
                "cached": False, "fetch_ms": 0.0, "status": "data_missing"
            }

        overall_score = 0.0
        genetics_score = 0.0
        evidence_count = 0

        for a in associations:
            if a.source == "opentargets":
                overall_score = max(overall_score, a.score)
            elif a.source == "opentargets_genetics":
                genetics_score = max(genetics_score, a.score)
            evidence_count += len(a.evidence or [])

        status = "ok" if (
                (overall_score > 0.0) or (genetics_score > 0.0) or (evidence_count > 0)
        ) else "data_missing"
        return {
            "overall": overall_score,
            "overallScore": overall_score,
            "genetics": genetics_score,
            "evidence_count": evidence_count,
            "release": "OT-25.06",
            "cached": True,
            "fetch_ms": 0.0,
            "status": status
        }

    except Exception as e:
        logger.error("Association fetch failed: %s", e, extra={"gene": target_symbol_or_ensg, "disease": disease_efo})
        return {
            "overall": 0.0, "genetics": 0.0,
            "evidence_count": 0, "release": "OT-unknown",
            "cached": False, "fetch_ms": 0.0, "status": "error", "error": str(e)
        }
