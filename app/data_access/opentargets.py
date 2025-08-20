"""
Open Targets GraphQL API client with caching + robust parsing.
Drop-in for fetch_ot_association(disease_efo, target_symbol_or_ensg).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

# ------------------------
# Cache configuration
# ------------------------
CACHE_DIR = Path(os.getenv("OT_CACHE_DIR", "cache/opentargets"))
CACHE_TTL_HOURS = int(os.getenv("OT_CACHE_TTL_HOURS", "48"))

# ------------------------
# HTTP configuration
# ------------------------
OT_GQL = os.getenv("OT_GRAPHQL_URL", "https://api.platform.opentargets.org/api/v4/graphql")
MAX_RETRIES = 2
RETRY_BACKOFF = 0.6  # seconds


# ========================
# Disk cache
# ========================
class OpenTargetsCache:
    """Simple disk cache for Open Targets API responses."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str, variables: Dict) -> str:
        payload = f"{query}:{json.dumps(variables, sort_keys=True, default=str)}"
        return hashlib.md5(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, query: str, variables: Dict) -> Optional[Dict]:
        key = self._key(query, variables)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                cached = json.load(f)
            ts = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - ts > timedelta(hours=CACHE_TTL_HOURS):
                path.unlink(missing_ok=True)
                return None
            return cached["data"]
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def set(self, query: str, variables: Dict, data: Dict) -> None:
        key = self._key(query, variables)
        path = self._path(key)
        payload = {"timestamp": datetime.now().isoformat(), "data": data}
        with open(path, "w") as f:
            json.dump(payload, f)


# ========================
# Client
# ========================
class OpenTargetsClient:
    """Open Targets GraphQL client with caching and retries."""

    def __init__(self, base_url: str = OT_GQL):
        self.base_url = base_url
        self.cache = OpenTargetsCache()
        self.session: Optional[httpx.AsyncClient] = None

        # Quick static fallbacks for most common interview genes
        self._static_symbol2ensg = {
            "EGFR": "ENSG00000146648",
            "ERBB2": "ENSG00000141736",
            "MET": "ENSG00000105976",
            "ALK": "ENSG00000171094",
            "KRAS": "ENSG00000133703",
            "BRAF": "ENSG00000157764",
            "PIK3CA": "ENSG00000121879",
            "ROS1": "ENSG00000112559",
            "RET": "ENSG00000165731",
            "NTRK1": "ENSG00000198400",
        }

    async def _get_session(self) -> httpx.AsyncClient:
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={"Content-Type": "application/json"},
                http2=True,
            )
        return self.session

    async def close(self) -> None:
        if self.session is not None:
            await self.session.aclose()
            self.session = None

    # ------------------------
    # Low-level query with cache + retries
    # ------------------------
    async def query(self, query: str, variables: Dict) -> Tuple[Dict, bool, float]:
        """
        Execute GraphQL query with disk cache.

        Returns:
            (json_response, was_cached, fetch_time_ms)
        """
        t0 = time.time()

        # Cache first
        cached = self.cache.get(query, variables)
        if cached:
            return cached, True, (time.time() - t0) * 1000

        # HTTP request with minimal retries
        client = await self._get_session()
        payload = {"query": query, "variables": variables}

        for attempt in range(1 + MAX_RETRIES):
            try:
                resp = await client.post(self.base_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data:
                    raise RuntimeError(f"GraphQL errors: {data['errors']}")
                self.cache.set(query, variables, data)
                return data, False, (time.time() - t0) * 1000
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    raise RuntimeError(f"Open Targets API error: {e}") from e
                await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))

        # Unreachable
        raise RuntimeError("Unexpected query failure")

    # ------------------------
    # Helpers
    # ------------------------
    async def get_target_id(self, gene_symbol_or_ensg: str) -> Optional[str]:
        """
        Resolve gene symbol to ENSG. If ENSG is already given, return as-is.
        """
        x = gene_symbol_or_ensg.strip().upper()
        if x.startswith("ENSG"):
            return x
        if x in self._static_symbol2ensg:
            return self._static_symbol2ensg[x]

        q = """
        query FindTarget($q:String!){
          search(queryString:$q, entityNames:["target"]){
            hits{ id name entity }
          }
        }"""
        variables = {"q": x}

        try:
            data, _, _ = await self.query(q, variables)
            hits = (data.get("data", {}) or {}).get("search", {}).get("hits", []) or []
            for h in hits:
                if h.get("entity") == "target" and h.get("name", "").upper() == x:
                    hid = h.get("id", "")
                    if hid.startswith("ENSG"):
                        return hid
            # fallback: first target hit with ENSG-like id
            for h in hits:
                hid = h.get("id", "")
                if hid.startswith("ENSG"):
                    return hid
        except Exception as e:
            print(f"[OT] get_target_id error for {x}: {e}")

        return None

    def _format_release(self, meta: Dict) -> str:
        try:
            dv = (meta or {}).get("dataVersion", {})
            y, m = dv.get("year"), dv.get("month")
            if isinstance(m, int):
                return f"OT-{y}.{m:02d}"
            return f"OT-{(meta or {}).get('version','unknown')}"
        except Exception:
            return "OT-unknown"

    # ------------------------
    # High-level: association scores
    # ------------------------
    async def get_association_scores(self, disease_efo: str, target_id: str) -> Dict:
        """
        Return a compact dict with association pieces we care about:
        genetics, text_mining, known_drug, overall, evidence_count, release, cached, fetch_ms
        """
        q = """
        query Assoc($d:String!, $t:String!){
          disease(efoId:$d){ id name }
          target(ensemblId:$t){ id approvedSymbol }
          associationDatasources(diseaseId:$d, targetId:$t){
            count
            rows{
              datasource{ id sectionName }
              score
              evidenceCount
            }
          }
          associationDiseaseTarget(diseaseId:$d, targetId:$t){
            id
            score
            datatypeScores{ id score }
          }
          meta{ version dataVersion{ year month iteration } }
        }"""
        variables = {"d": disease_efo, "t": target_id}
        data, cached, fetch_ms = await self.query(q, variables)

        # Prepare defaults
        out = {
            "overall": 0.0,
            "genetics": 0.0,
            "text_mining": 0.0,
            "known_drug": 0.0,
            "evidence_count": 0,
            "release": "OT-unknown",
            "cached": cached,
            "fetch_ms": fetch_ms,
        }

        # Release
        meta = (data.get("data", {}) or {}).get("meta", {})
        out["release"] = self._format_release(meta)

        # Prefer associationDiseaseTarget for overall/genetics (datatypeScores)
        adt = (data.get("data", {}) or {}).get("associationDiseaseTarget", {}) or {}
        if adt:
            try:
                out["overall"] = float(adt.get("score") or 0.0)
            except Exception:
                pass
            for ds in adt.get("datatypeScores", []) or []:
                dsid = (ds.get("id") or "").lower()
                if dsid in ("genetic_association", "genetics", "genetic"):
                    try:
                        out["genetics"] = max(out["genetics"], float(ds.get("score") or 0.0))
                    except Exception:
                        pass

        # Fallback/augment from associationDatasources
        ads = (data.get("data", {}) or {}).get("associationDatasources", {}) or {}
        rows = ads.get("rows", []) or []
        section_scores: Dict[str, Dict[str, float]] = {}
        total_evid = 0

        for row in rows:
            section = ((row.get("datasource", {}) or {}).get("sectionName") or "").lower()
            score = float(row.get("score") or 0.0)
            evid = int(row.get("evidenceCount") or 0)

            total_evid += evid
            if section not in section_scores:
                section_scores[section] = {"score": 0.0, "evidence": 0}
            section_scores[section]["score"] = max(section_scores[section]["score"], score)
            section_scores[section]["evidence"] += evid

        out["evidence_count"] = total_evid

        # map common sections
        def sec(name: str) -> float:
            return float(section_scores.get(name, {}).get("score", 0.0))

        # genetics fallback if ADT empty
        if out["genetics"] == 0.0:
            out["genetics"] = max(sec("genetic_association"), sec("genetics"))

        # literature / text-mining
        out["text_mining"] = max(sec("literature"), sec("text_mining"))

        # known drug
        out["known_drug"] = max(sec("drugs"), sec("known_drug"))

        # overall fallback if needed
        if out["overall"] == 0.0 and section_scores:
            out["overall"] = max(v["score"] for v in section_scores.values())

        return out


# ========================
# Global instance + helpers
# ========================
_ot_client: Optional[OpenTargetsClient] = None


async def get_ot_client() -> OpenTargetsClient:
    global _ot_client
    if _ot_client is None:
        _ot_client = OpenTargetsClient()
    return _ot_client


async def fetch_ot_association(disease_efo: str, target_symbol_or_ensg: str) -> Dict:
    """
    High-level convenience used by the rest of the app.
    Returns a compact dict with association scores and metadata.
    """
    client = await get_ot_client()
    ensg = (
        target_symbol_or_ensg
        if str(target_symbol_or_ensg).upper().startswith("ENSG")
        else await client.get_target_id(target_symbol_or_ensg)
    )
    if not ensg:
        return {
            "overall": 0.0,
            "genetics": 0.0,
            "text_mining": 0.0,
            "known_drug": 0.0,
            "evidence_count": 0,
            "release": "OT-unknown",
            "cached": False,
            "fetch_ms": 0.0,
            "error": f"Target not found: {target_symbol_or_ensg}",
        }

    return await client.get_association_scores(disease_efo, ensg)


async def cleanup_ot_client() -> None:
    global _ot_client
    if _ot_client is not None:
        await _ot_client.close()
        _ot_client = None
