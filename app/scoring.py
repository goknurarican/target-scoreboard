# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Target scoring implementation - Phase 1C Pipeline Architecture.
TargetBuilder async pipeline with validation, evidence tracking, and lineage.
"""
import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import dirichlet, kendalltau

from .schemas import (
    TargetScore,
    TargetBreakdown,
    ScoreRequest,
    ChannelScore,
    TargetScoreBundle,
    DataQualityFlags,
    EvidenceRef,
    get_utc_now
)
from .validation import get_validator

# Import legacy components for compatibility
from .channels.genetics import get_genetics_channel
from .channels.ppi_proximity import compute_ppi_proximity
from .channels.pathway import compute_pathway_enrichment
from .channels.modality_fit import compute_modality_fit
from .channels.safety import compute_safety_penalty
from .data_access.opentargets import fetch_ot_association

logger = logging.getLogger(__name__)

# Default weights configuration from environment
DEFAULT_WEIGHTS = {
    "genetics": float(os.getenv("WEIGHT_GENETICS", "0.35")),
    "ppi": float(os.getenv("WEIGHT_PPI", "0.25")),
    "pathway": float(os.getenv("WEIGHT_PATHWAY", "0.20")),
    "safety": float(os.getenv("WEIGHT_SAFETY", "0.10")),
    "modality_fit": float(os.getenv("WEIGHT_MODALITY", "0.10"))
}


class TargetBuilder:
    """
    Production target scoring pipeline with async data fetching and validation.

    Pipeline stages:
    1. fetch_all() - Parallel channel data fetching
    2. validate() - Quality validation per channel
    3. compute_scores() - Weight-based score combination
    4. assemble() - TargetScoreBundle with lineage tracking
    """

    def __init__(self, gene: str, disease: Optional[str] = None):
        self.gene = gene.upper().strip()
        self.disease = disease
        self.validator = get_validator()
        self.timestamp = get_utc_now()

        # Pipeline state
        self.raw_data: Dict[str, Any] = {}
        self.validated_data: Dict[str, ChannelScore] = {}
        self.lineage: Dict[str, Any] = {
            "gene": self.gene,
            "disease": self.disease,
            "pipeline_version": "v1.0.0-phase1c",
            "timestamp": self.timestamp.isoformat(),
            "sources": [],
            "transforms": [],
            "quality_issues": []
        }

    async def fetch_all(self) -> Dict[str, Any]:
        """
        Fetch data from all channels in parallel.

        Returns:
            Dict with raw channel data
        """
        start_time = time.time()

        try:
            # Create async tasks for each channel
            tasks = {}

            if self.disease:
                # Genetics channel (requires disease)
                tasks["genetics"] = self._fetch_genetics_data()

            # Other channels
            tasks["ppi"] = self._fetch_ppi_data()
            tasks["pathway"] = self._fetch_pathway_data()
            tasks["safety"] = self._fetch_safety_data()
            tasks["modality_fit"] = self._fetch_modality_data()

            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Map results back to channel names
            for channel, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Channel {channel} fetch failed: {result}")
                    self.raw_data[channel] = {"status": "error", "error": str(result)}
                    self.lineage["quality_issues"].append(f"{channel}: {str(result)[:100]}")
                else:
                    self.raw_data[channel] = result
                    self.lineage["sources"].append(f"{channel}: fetched")

            fetch_time = (time.time() - start_time) * 1000
            self.lineage["transforms"].append(f"fetch_all: {fetch_time:.1f}ms")

            logger.info(
                f"Data fetching completed for {self.gene}",
                extra={
                    "gene": self.gene,
                    "disease": self.disease,
                    "channels_fetched": len([k for k, v in self.raw_data.items() if "error" not in v]),
                    "fetch_time_ms": fetch_time
                }
            )

            return self.raw_data

        except Exception as e:
            logger.error(f"Parallel data fetch failed for {self.gene}: {e}")
            self.lineage["quality_issues"].append(f"fetch_all_error: {str(e)}")
            raise


    def _normalize_disease(self) -> tuple[str, str | None]:
        import re

        EFO_RE = re.compile(r"^[A-Z]+_[0-9]{4,}$")

        """Return (disease_id, note) with best-effort mapping."""
        if self.disease and EFO_RE.match(self.disease):
            return self.disease, None

        # minimal mapping; genişletebilirsin
        mapping = {
            "non-small cell lung carcinoma": "EFO_0003071",
            "nsclc": "EFO_0003071",
            "lung cancer": "EFO_0000305",
        }
        if not self.disease:
            return "EFO_0000305", "Disease missing → defaulted to EFO_0000305"

        key = self.disease.strip().lower()
        if key in mapping:
            return mapping[key], f"Mapped '{self.disease}' → {mapping[key]}"

        # bilinmiyorsa olduğu gibi dön (kanal yine data_missing üretebilir)
        return self.disease, "Non-ontology disease string; channels may be partial"

    async def _fetch_genetics_data(self) -> ChannelScore:
        """Fetch genetics channel data with OT fallback + disease normalization."""
        if not self.disease:
            return ChannelScore(
                name="genetics",
                score=None,
                status="data_missing",
                components={},
                evidence=[],
                quality=DataQualityFlags(notes="No disease provided for genetics channel"),
            )

        disease_id, note = self._normalize_disease()

        cs: Optional[ChannelScore] = None

        # 1) Ana kanal
        try:
            genetics_channel = await get_genetics_channel()
            raw = await genetics_channel.compute_score(self.gene, disease_id)
            if isinstance(raw, ChannelScore):
                cs = raw
            else:
                cs = ChannelScore(
                    name="genetics",
                    score=raw.get("score"),
                    status=raw.get("status", "ok" if raw.get("score") is not None else "data_missing"),
                    components=raw.get("components", {}),
                    evidence=self._convert_evidence_refs(raw.get("evidence_refs", [])),
                )
        except Exception as e:
            logger.error(f"Genetics compute failed for {self.gene}: {e}")
            cs = ChannelScore(
                name="genetics",
                score=None,
                status="error",
                components={"error": str(e)[:100]},
                evidence=[],
                quality=DataQualityFlags(notes=f"Exception: {str(e)[:100]}"),
            )

        # 2) Fallback: OpenTargets association skorunu kullan
        try:
            # 0.0 skoru da fallback sebebidir
            needs_fallback = (cs is None) or (cs.status != "ok") or (float(cs.score or 0.0) <= 0.0)

            def _extract_ot_score(obj: Any) -> Optional[float]:
                if not isinstance(obj, dict):
                    return None
                # Sık görülen isimler
                for k in ("overall", "overallScore", "association_score", "score"):
                    v = obj.get(k)
                    if isinstance(v, dict):
                        for kk in ("overall", "value", "score"):
                            if kk in v:
                                v = v[kk]
                                break
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
                # Derin yollar
                for path in [
                    ("data", "association_score", "overall"),
                    ("association_score", "overall"),
                    ("scores", "overall"),
                ]:
                    cur = obj
                    ok = True
                    for p in path:
                        if isinstance(cur, dict) and p in cur:
                            cur = cur[p]
                        else:
                            ok = False
                            break
                    if ok:
                        try:
                            return float(cur)
                        except Exception:
                            pass
                return None

            if needs_fallback:
                ot = await fetch_ot_association(disease_id, self.gene)
                ot_score = _extract_ot_score(ot)
                if ot_score is not None:
                    ot_score = max(0.0, min(1.0, float(ot_score)))
                    if cs is None:
                        cs = ChannelScore(name="genetics", score=None, status="data_missing", components={},
                                          evidence=[])
                    cs.score = ot_score
                    cs.status = "ok"
                    cs.components = {**(cs.components or {}), "fallback_opentargets": ot_score}
                    cs.evidence = [
                        *cs.evidence,
                        EvidenceRef(
                            source="opentargets",
                            title="OpenTargets association (fallback)",
                            url="https://platform.opentargets.org/",
                            timestamp=get_utc_now(),
                        ),
                    ]
                    cs.quality = cs.quality or DataQualityFlags()
                    cs.quality.partial = True
                    cs.quality.notes = ((
                                                    cs.quality.notes + " | ") if cs.quality.notes else "") + "Genetics via OT fallback"
        except Exception as e:
            logger.warning(f"Genetics OT fallback failed: {e}")

        # 3) Not bilgisini kaliteye işle
        if note:
            cs.quality = cs.quality or DataQualityFlags()
            cs.quality.partial = True
            cs.quality.notes = (cs.quality.notes + " | " if cs.quality.notes else "") + note

        # Son güvenlik: hiç oluşturulamadıysa data_missing döndür
        if cs is None:
            cs = ChannelScore(
                name="genetics",
                score=None,
                status="data_missing",
                components={},
                evidence=[],
                quality=DataQualityFlags(notes="Genetics channel produced no result"),
            )

        logger.info(f"[GENETICS] {self.gene} {disease_id} score={cs.score} status={cs.status}")
        return cs

    async def _fetch_ppi_data(self) -> Dict[str, Any]:
        """Fetch PPI proximity data + lightweight neighbors for UI."""
        try:
            # 1) Hastalık bağlamından seed genler
            disease_genes: List[str] = []
            if self.disease:
                major_cancer_genes = ["TP53", "EGFR", "KRAS", "PIK3CA", "APC", "BRCA1", "BRCA2", "MYC"]
                disease_genes = [g for g in major_cancer_genes if g != self.gene][:5]

            # 2) PPI yakınlığı hesapla
            ppi_score, ppi_refs = await compute_ppi_proximity(
                self.gene, disease_genes=disease_genes, rwr_enabled=True
            )

            # 3) Evidence metinlerinden komşu çıkar (STRING formatlarına toleranslı)
            neighbors: List[Dict[str, Any]] = []
            import re
            gene_up = self.gene.upper()

            for r in (ppi_refs or []):
                try:
                    st = str(r)
                    # Örnekler: "EGFR - KRAS (0.89)", "EGFR–KRAS 0.82", "EGFR-KRAS score=0.77"
                    m = re.search(
                        r"([A-Z0-9-]{2,})\s*[–\-]\s*([A-Z0-9-]{2,}).*?([01](?:\.\d+)?|\.\d+)",
                        st
                    )
                    if m:
                        a, b, conf = m.groups()
                        partner = b if a.upper() == gene_up else a
                        neighbors.append({
                            "partner": partner.upper(),
                            "confidence": float(conf),
                            "source": "stringdb"
                        })
                except Exception:
                    continue

            # 4) Hiç komşu çıkmadıysa bağlam genlerinden hafif fallback komşular üret
            if not neighbors and disease_genes:
                neighbors = [{"partner": g, "confidence": 0.30, "source": "fallback"} for g in disease_genes]

            # 5) Bayraklar
            error_flag = any(isinstance(r, str) and r.lower().startswith("status:error") for r in (ppi_refs or []))
            used_fallback = any(isinstance(r, str) and "source:fallback" in r.lower() for r in (ppi_refs or []))

            # 6) UI bileşenleri
            components: Dict[str, Any] = {
                "seed_genes_in_context": float(len(disease_genes)),
                "rwr_enabled": 1.0,
                "neighbors": neighbors[:12],  # UI için yeter
                "graph_hint": bool(neighbors),  # Network kartını açmaya yardımcı
            }
            if used_fallback:
                components["used_fallback"] = 1.0

            # 7) Durum belirleme ve pseudo-skor
            # Skor yoksa ama komşu varsa, UI ağ kartı açılsın diye minik bir pseudo-skor ver
            if ppi_score is None and neighbors:
                ppi_score = 1e-3
            status = "ok" if (ppi_score is not None and ppi_score >= 0.0) or neighbors else "data_missing"

            # Sadece gerçek hata ve hiç komşu yoksa 'error' döndür
            if error_flag and not neighbors:
                return {
                    "score": None,
                    "evidence_refs": ppi_refs,
                    "status": "error",
                    "components": components
                }

            out = {
                "score": ppi_score,
                "evidence_refs": ppi_refs,
                "status": status,
                "components": components,
            }
            if ppi_score == 1e-3:
                out["quality_note"] = "PPI via neighbors-only (pseudo-score)"

            # Tanı için kısa log
            try:
                logger.info(f"[PPI] {self.gene} score={ppi_score} neighbors={len(neighbors)} status={status}")
            except Exception:
                pass

            return out

        except Exception as e:
            logger.error(f"PPI data fetch failed for {self.gene}: {e}")
            return {"status": "error", "error": str(e), "components": {}}

    # Copyright (c) 2025 Göknur Arıcan
    # All rights reserved. Licensed for internal evaluation only.
    # See LICENSE-EVALUATION.md for terms.

    """
    Target scoring implementation - Phase 1C Pipeline Architecture.
    TargetBuilder async pipeline with validation, evidence tracking, and lineage.
    """
    import asyncio
    import logging
    import os
    import time
    from datetime import datetime
    from typing import Dict, List, Optional, Tuple, Any

    import numpy as np
    from scipy.stats import dirichlet, kendalltau

    from .schemas import (
        TargetScore,
        TargetBreakdown,
        ScoreRequest,
        ChannelScore,
        TargetScoreBundle,
        DataQualityFlags,
        EvidenceRef,
        get_utc_now
    )
    from .validation import get_validator

    # Import legacy components for compatibility
    from .channels.genetics import get_genetics_channel
    from .channels.ppi_proximity import compute_ppi_proximity
    from .channels.pathway import compute_pathway_enrichment
    from .channels.modality_fit import compute_modality_fit
    from .channels.safety import compute_safety_penalty
    from .data_access.opentargets import fetch_ot_association

    logger = logging.getLogger(__name__)

    # Default weights configuration from environment
    DEFAULT_WEIGHTS = {
        "genetics": float(os.getenv("WEIGHT_GENETICS", "0.35")),
        "ppi": float(os.getenv("WEIGHT_PPI", "0.25")),
        "pathway": float(os.getenv("WEIGHT_PATHWAY", "0.20")),
        "safety": float(os.getenv("WEIGHT_SAFETY", "0.10")),
        "modality_fit": float(os.getenv("WEIGHT_MODALITY", "0.10"))
    }

    class TargetBuilder:
        """
        Production target scoring pipeline with async data fetching and validation.

        Pipeline stages:
        1. fetch_all() - Parallel channel data fetching
        2. validate() - Quality validation per channel
        3. compute_scores() - Weight-based score combination
        4. assemble() - TargetScoreBundle with lineage tracking
        """

        def __init__(self, gene: str, disease: Optional[str] = None):
            self.gene = gene.upper().strip()
            self.disease = disease
            self.validator = get_validator()
            self.timestamp = get_utc_now()

            # Pipeline state
            self.raw_data: Dict[str, Any] = {}
            self.validated_data: Dict[str, ChannelScore] = {}
            self.lineage: Dict[str, Any] = {
                "gene": self.gene,
                "disease": self.disease,
                "pipeline_version": "v1.0.0-phase1c",
                "timestamp": self.timestamp.isoformat(),
                "sources": [],
                "transforms": [],
                "quality_issues": []
            }

        async def fetch_all(self) -> Dict[str, Any]:
            """
            Fetch data from all channels in parallel.

            Returns:
                Dict with raw channel data
            """
            start_time = time.time()

            try:
                # Create async tasks for each channel
                tasks = {}

                if self.disease:
                    # Genetics channel (requires disease)
                    tasks["genetics"] = self._fetch_genetics_data()

                # Other channels
                tasks["ppi"] = self._fetch_ppi_data()
                tasks["pathway"] = self._fetch_pathway_data()
                tasks["safety"] = self._fetch_safety_data()
                tasks["modality_fit"] = self._fetch_modality_data()

                # Execute all tasks in parallel
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)

                # Map results back to channel names
                for channel, result in zip(tasks.keys(), results):
                    if isinstance(result, Exception):
                        logger.error(f"Channel {channel} fetch failed: {result}")
                        self.raw_data[channel] = {"status": "error", "error": str(result)}
                        self.lineage["quality_issues"].append(f"{channel}: {str(result)[:100]}")
                    else:
                        self.raw_data[channel] = result
                        self.lineage["sources"].append(f"{channel}: fetched")

                fetch_time = (time.time() - start_time) * 1000
                self.lineage["transforms"].append(f"fetch_all: {fetch_time:.1f}ms")

                logger.info(
                    f"Data fetching completed for {self.gene}",
                    extra={
                        "gene": self.gene,
                        "disease": self.disease,
                        "channels_fetched": len([k for k, v in self.raw_data.items() if "error" not in v]),
                        "fetch_time_ms": fetch_time
                    }
                )

                return self.raw_data

            except Exception as e:
                logger.error(f"Parallel data fetch failed for {self.gene}: {e}")
                self.lineage["quality_issues"].append(f"fetch_all_error: {str(e)}")
                raise

        def _normalize_disease(self) -> tuple[str, str | None]:
            import re

            EFO_RE = re.compile(r"^[A-Z]+_[0-9]{4,}$")

            """Return (disease_id, note) with best-effort mapping."""
            if self.disease and EFO_RE.match(self.disease):
                return self.disease, None

            # minimal mapping; genişletebilirsin
            mapping = {
                "non-small cell lung carcinoma": "EFO_0003071",
                "nsclc": "EFO_0003071",
                "lung cancer": "EFO_0000305",
            }
            if not self.disease:
                return "EFO_0000305", "Disease missing → defaulted to EFO_0000305"

            key = self.disease.strip().lower()
            if key in mapping:
                return mapping[key], f"Mapped '{self.disease}' → {mapping[key]}"

            # bilinmiyorsa olduğu gibi dön (kanal yine data_missing üretebilir)
            return self.disease, "Non-ontology disease string; channels may be partial"

        async def _fetch_genetics_data(self) -> ChannelScore:
            """Fetch genetics channel data with OT fallback + disease normalization."""
            if not self.disease:
                return ChannelScore(
                    name="genetics",
                    score=None,
                    status="data_missing",
                    components={},
                    evidence=[],
                    quality=DataQualityFlags(notes="No disease provided for genetics channel"),
                )

            disease_id, note = self._normalize_disease()

            # 1) Ana kanal
            try:
                genetics_channel = await get_genetics_channel()
                cs = await genetics_channel.compute_score(self.gene, disease_id)
                # bazı implementasyonlar dict döndürebiliyor; güvenceye al
                if not isinstance(cs, ChannelScore):
                    cs = ChannelScore(
                        name="genetics",
                        score=cs.get("score"),
                        status=cs.get("status", "ok" if cs.get("score") is not None else "data_missing"),
                        components=cs.get("components", {}),
                        evidence=self._convert_evidence_refs(cs.get("evidence_refs", [])),
                    )
            except Exception as e:
                logger.error(f"Genetics compute failed for {self.gene}: {e}")
                cs = ChannelScore(
                    name="genetics",
                    score=None,
                    status="error",
                    components={"error": str(e)[:100]},
                    evidence=[],
                    quality=DataQualityFlags(notes=f"Exception: {str(e)[:100]}"),
                )

            # 2) Fallback: OpenTargets association skorunu kullan
            # Fallback: OpenTargets
            try:
                needs_fallback = (cs.score is None) or (cs.status != "ok")
                if needs_fallback:
                    ot = await fetch_ot_association(disease_id, self.gene)

                    ok_signal = (
                            ot.get("status") == "ok" and
                            (
                                    float(ot.get("overallScore", ot.get("overall", 0.0)) or 0.0) > 0.0 or
                                    float(ot.get("genetics", 0.0) or 0.0) > 0.0 or
                                    int(ot.get("evidence_count", 0) or 0) > 0
                            )
                    )

                    if ok_signal:
                        ot_score = float(ot.get("overallScore", ot.get("overall", 0.0)) or 0.0)
                        cs.score = ot_score
                        cs.status = "ok"
                        cs.components = {**(cs.components or {}), "fallback_opentargets": ot_score}
                        cs.evidence = [
                            *cs.evidence,
                            EvidenceRef(source="opentargets",
                                        title="OpenTargets association (fallback)",
                                        url="https://platform.opentargets.org/")
                        ]
                        cs.quality = cs.quality or DataQualityFlags()
                        cs.quality.partial = True
                        cs.quality.notes = ((
                                                        cs.quality.notes + " | ") if cs.quality.notes else "") + "Genetics via OT fallback"
                    else:
                        # Fallback da sinyal getirmediyse: 0'ı 'ok' diye göstermeyelim
                        cs.score = None
                        cs.status = "data_missing"
                        cs.components = {**(cs.components or {}), "fallback_opentargets": 0.0}
                        cs.quality = cs.quality or DataQualityFlags()
                        cs.quality.partial = True
                        cs.quality.notes = ((
                                                        cs.quality.notes + " | ") if cs.quality.notes else "") + "No associations found"
            except Exception as e:
                logger.warning(f"Genetics OT fallback failed: {e}")

            # 3) Not bilgisini kaliteye işle
            if note:
                cs.quality = cs.quality or DataQualityFlags()
                cs.quality.partial = True
                cs.quality.notes = (cs.quality.notes + " | " if cs.quality.notes else "") + note

            return cs

        async def _fetch_ppi_data(self) -> Dict[str, Any]:
            """Fetch PPI proximity data + lightweight neighbors for UI."""
            try:
                disease_genes = []
                if self.disease:
                    major_cancer_genes = ["TP53", "EGFR", "KRAS", "PIK3CA", "APC", "BRCA1", "BRCA2", "MYC"]
                    disease_genes = [g for g in major_cancer_genes if g != self.gene][:5]

                ppi_score, ppi_refs = await compute_ppi_proximity(
                    self.gene, disease_genes=disease_genes, rwr_enabled=True
                )

                # refs'ten kaba komşu çıkart (STRING tarzı metinler için esnek parse)
                neighbors = []
                import re
                for r in (ppi_refs or []):
                    try:
                        st = str(r)
                        m = re.search(r"([A-Z0-9-]{2,})\s*[-–]\s*([A-Z0-9-]{2,}).*?(0\.\d+|1(?:\.0+)?)", st)
                        if m:
                            a, b, conf = m.groups()
                            partner = b if a == self.gene else a
                            neighbors.append({"partner": partner, "confidence": float(conf), "source": "stringdb"})
                    except Exception:
                        continue

                # hiçbir şey parse edemediysek bağlam genlerinden dummy komşular ekle
                if not neighbors and disease_genes:
                    neighbors = [{"partner": g, "confidence": 0.3, "source": "fallback"} for g in disease_genes]

                error_flag = any(isinstance(r, str) and r.startswith("Status:error") for r in (ppi_refs or []))
                used_fallback = any(isinstance(r, str) and "Source:fallback" in r for r in (ppi_refs or []))

                components = {
                    "seed_genes_in_context": float(len(disease_genes)),
                    "rwr_enabled": 1.0,
                    "neighbors": neighbors[:12],  # UI için yeter
                }

                if error_flag:
                    return {"score": None, "evidence_refs": ppi_refs, "status": "error", "components": components}

                status = "ok" if (ppi_score is not None and ppi_score >= 0) else "data_missing"
                if used_fallback:
                    components["used_fallback"] = 1.0
                    # fallback olsa da UI ağ gösterebilsin diye "ok" tut
                    status = "ok"

                return {"score": ppi_score, "evidence_refs": ppi_refs, "status": status, "components": components}

            except Exception as e:
                logger.error(f"PPI data fetch failed for {self.gene}: {e}")
                return {"status": "error", "error": str(e), "components": {}}

        async def _fetch_pathway_data(self) -> Dict[str, Any]:
            """Fetch pathway enrichment data."""
            try:
                targets_context = [self.gene]  # Simplified
                pathway_score, pathway_refs = await compute_pathway_enrichment(self.gene, targets_context)

                return {
                    "score": pathway_score,
                    "evidence_refs": pathway_refs,
                    "status": "ok" if pathway_score > 0 else "data_missing"
                }
            except Exception as e:
                logger.error(f"Pathway data fetch failed for {self.gene}: {e}")
                return {"status": "error", "error": str(e)}

        async def _fetch_safety_data(self) -> Dict[str, Any]:
            try:
                disease_id, note = self._normalize_disease()
                ot_data = await fetch_ot_association(disease_id, self.gene)
                safety_score, safety_refs = await compute_safety_penalty(self.gene, ot_data)
                status = "ok" if safety_score is not None else "data_missing"
                out = {"score": safety_score, "evidence_refs": safety_refs, "status": status}
                if note: out["quality_note"] = note
                return out
            except Exception as e:
                logger.error(f"Safety data fetch failed for {self.gene}: {e}")
                return {"status": "error", "error": str(e)}

        async def _fetch_modality_data(self) -> Dict[str, Any]:
            """Fetch modality fit data."""
            try:
                modality_scores, modality_refs = await compute_modality_fit(self.gene, None)

                return {
                    "scores": modality_scores,
                    "evidence_refs": modality_refs,
                    "status": "ok" if modality_scores else "data_missing"
                }
            except Exception as e:
                logger.error(f"Modality data fetch failed for {self.gene}: {e}")
                return {"status": "error", "error": str(e)}

        async def validate(self, raw_data: Dict[str, Any]) -> Dict[str, ChannelScore]:
            validated = {}
            validation_start = time.time()

            for channel, data in raw_data.items():
                try:
                    # Genetics zaten ChannelScore dönebiliyor → doğrudan geçir
                    if channel == "genetics" and isinstance(data, ChannelScore):
                        validated[channel] = data
                        continue

                    # Gelen yapıyı bozmadan alanları çek
                    if isinstance(data, dict):
                        incoming_status = data.get("status")
                        score = data.get("score")
                        evidence_refs = self._convert_evidence_refs(data.get("evidence_refs", []))
                        components = data.get("components", {})
                    else:
                        incoming_status = None
                        score = None
                        evidence_refs = []
                        components = {}

                    # Modality özel: içteki skorları components’e geçir ve overall’u skor olarak kullan
                    if channel == "modality_fit" and isinstance(data, dict) and "scores" in data:
                        m = data["scores"]
                        if isinstance(m, dict):
                            components.update(m)
                            score = m.get("overall_druggability", score)
                        elif hasattr(m, "overall_druggability"):
                            score = getattr(m, "overall_druggability", score)

                    # Yalnızca kalite bayrakları için validator’ı çağır (veriyi EZME)
                    vres = self.validator.validate(channel, data if isinstance(data, (dict, list)) else {})

                    # Statüyü koru; yoksa skora göre üret
                    if incoming_status in {"ok", "data_missing", "error"}:
                        status = incoming_status
                    else:
                        status = "ok" if (score is not None) else "data_missing"

                    validated[channel] = ChannelScore(
                        name=channel,
                        score=score,
                        status=status,
                        components=components,  # <-- KORU (neighbors burada kalır)
                        evidence=evidence_refs,  # <-- KORU
                        quality=vres.quality
                    )

                except Exception as e:
                    logger.error(f"Validation failed for {channel}: {e}")
                    validated[channel] = ChannelScore(
                        name=channel,
                        score=None,
                        status="error",
                        components={"validation_error": str(e)[:100]},
                        evidence=[],
                        quality=DataQualityFlags(notes=f"Validation error: {str(e)[:100]}")
                    )

            validation_time = (time.time() - validation_start) * 1000
            self.lineage["transforms"].append(f"validate: {validation_time:.1f}ms")
            return validated

        def _convert_evidence_refs(self, evidence_strings: List[str]) -> List[EvidenceRef]:
            """Convert legacy evidence strings like 'Source:stringdb' into structured EvidenceRef."""
            out: List[EvidenceRef] = []
            for s in evidence_strings:
                try:
                    st = s.strip()
                    if st.startswith("PMID:"):
                        pmid = st.split("PMID:")[1].split()[0]
                        out.append(EvidenceRef(source="pubmed",
                                               pmid=pmid,
                                               url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                               timestamp=get_utc_now()))
                        continue
                    if st.lower().startswith("source:"):
                        src = st.split(":", 1)[1].strip().lower()
                        if src in {"stringdb", "string"}:
                            out.append(EvidenceRef(source="stringdb",
                                                   title="STRING evidence",
                                                   url="https://string-db.org/",
                                                   timestamp=get_utc_now()))
                        elif src in {"opentargets", "ot"}:
                            out.append(EvidenceRef(source="opentargets",
                                                   title="OpenTargets evidence",
                                                   url="https://platform.opentargets.org/",
                                                   timestamp=get_utc_now()))
                        elif src in {"fallback"}:
                            out.append(EvidenceRef(source="fallback",
                                                   title="Fallback evidence",
                                                   timestamp=get_utc_now()))
                        else:
                            out.append(EvidenceRef(source=src or "unknown",
                                                   title=st[:100],
                                                   timestamp=get_utc_now()))
                        continue
                    if "STRING:" in st:
                        out.append(EvidenceRef(source="stringdb",
                                               title=st[:100],
                                               url="https://string-db.org/",
                                               timestamp=get_utc_now()))
                    elif "OpenTargets:" in st or "OT-" in st:
                        out.append(EvidenceRef(source="opentargets",
                                               title=st[:100],
                                               url="https://platform.opentargets.org/",
                                               timestamp=get_utc_now()))
                    else:
                        out.append(EvidenceRef(source="unknown",
                                               title=st[:100],
                                               timestamp=get_utc_now()))
                except Exception as e:
                    logger.warning(f"Evidence parse error: {s} ({e})")
            return out

        async def compute_scores(self, validated_data: Dict[str, ChannelScore], weights: Dict[str, float]) -> Dict[
            str, float]:
            """
            Combine channel scores using config-driven weights.

            Args:
                validated_data: Validated channel data
                weights: Channel weights configuration

            Returns:
                Dict with final scores and metadata
            """
            compute_start = time.time()

            # Extract scores from validated channels
            channel_scores = {}
            available_channels = []
            total_weight = 0.0

            for channel, channel_data in validated_data.items():
                if channel_data.status == "ok" and channel_data.score is not None:
                    score = float(channel_data.score)

                    # Handle safety inversion (safety is penalty)
                    if channel == "safety":
                        score = 1.0 - score

                    # Ensure bounds
                    score = max(0.0, min(1.0, score))
                    channel_scores[channel] = score
                    available_channels.append(channel)

                    if channel in weights:
                        total_weight += weights[channel]

            # Normalize weights for available channels
            if total_weight > 0:
                normalized_weights = {
                    ch: weights.get(ch, 0.0) / total_weight
                    for ch in available_channels
                }
            else:
                # All channels failed - use equal weights
                normalized_weights = {ch: 1.0 / len(available_channels) for ch in available_channels}

            # Compute weighted total score
            combined_score = 0.0
            for channel, score in channel_scores.items():
                weight = normalized_weights.get(channel, 0.0)
                combined_score += score * weight

            # Ensure minimum score
            combined_score = max(0.1, min(1.0, combined_score))

            compute_time = (time.time() - compute_start) * 1000
            self.lineage["transforms"].append(f"compute_scores: {compute_time:.1f}ms")

            logger.info(
                f"Scores computed for {self.gene}",
                extra={
                    "gene": self.gene,
                    "combined_score": combined_score,
                    "available_channels": available_channels,
                    "total_weight": total_weight
                }
            )

            return {
                "combined_score": combined_score,
                "channel_scores": channel_scores,
                "normalized_weights": normalized_weights,
                "available_channels": available_channels
            }

        async def assemble(self, weights: Dict[str, float]) -> TargetScoreBundle:
            """
            Assemble complete TargetScoreBundle with lineage tracking.

            Args:
                weights: Channel weights configuration

            Returns:
                Complete TargetScoreBundle
            """
            # Execute pipeline stages
            raw_data = await self.fetch_all()
            validated_data = await self.validate(raw_data)
            score_results = await self.compute_scores(validated_data, weights)

            # Build final bundle
            bundle = TargetScoreBundle(
                gene=self.gene,
                disease=self.disease,
                channels=validated_data,
                combined_score=score_results["combined_score"],
                lineage=self.lineage,
                timestamp=self.timestamp
            )

            # Update lineage with final results
            self.lineage["final_score"] = score_results["combined_score"]
            self.lineage["available_channels"] = score_results["available_channels"]
            self.lineage["pipeline_completed"] = get_utc_now().isoformat()

            return bundle

    # ========================
    # Legacy compatibility layer for existing API
    # ========================

    class LegacyTargetScorer:
        """Legacy scorer for backward compatibility with existing main.py."""

        def __init__(self):
            self.default_weights = DEFAULT_WEIGHTS

        async def score_single_target(
                self,
                disease: str,
                target: str,
                weights: Dict[str, float],
                targets_context: List[str] = None,
                version_manager=None
        ) -> TargetScore:
            """Legacy single target scoring with TargetBuilder pipeline."""

            # Use new TargetBuilder pipeline
            builder = TargetBuilder(target, disease)
            bundle = await builder.assemble(weights)
            channels_export = {name: cs.model_dump() for name, cs in bundle.channels.items()}

            # Convert TargetScoreBundle to legacy TargetScore format
            breakdown = TargetBreakdown()
            all_evidence_refs = []

            # Extract scores from channels
            for channel_name, channel_score in bundle.channels.items():
                if channel_name == "genetics":
                    breakdown.genetics = channel_score.score
                elif channel_name == "ppi":
                    breakdown.ppi_proximity = channel_score.score
                elif channel_name == "pathway":
                    breakdown.pathway_enrichment = channel_score.score
                elif channel_name == "safety":
                    breakdown.safety_off_tissue = channel_score.score
                elif channel_name == "modality_fit":
                    breakdown.modality_fit = channel_score.components

                # Collect evidence references
                for evidence in channel_score.evidence:
                    if evidence.pmid:
                        all_evidence_refs.append(f"PMID:{evidence.pmid}")
                    all_evidence_refs.append(f"Source:{evidence.source}")

            # Build explanation (simplified for compatibility)
            explanation = {
                "target": target,
                "total_weighted_score": bundle.combined_score or 0.0,
                "confidence_level": "medium",
                "key_insights": [
                    f"Scored using TargetBuilder pipeline v{bundle.lineage.get('pipeline_version', '1.0')}"]
            }

            # Data version from lineage
            data_version = f"Pipeline-{bundle.lineage.get('pipeline_version', 'unknown')}"

            return TargetScore(
                target=target,
                total_score=bundle.combined_score or 0.0,
                breakdown=breakdown,
                evidence_refs=list(dict.fromkeys(all_evidence_refs)),  # Remove duplicates
                data_version=data_version,
                explanation=explanation,
                timestamp=bundle.timestamp,
                warnings=None,
                channels=bundle.channels,
            )

        async def score_targets_batch(self, request: ScoreRequest) -> Tuple[List[TargetScore], Dict]:
            """Legacy batch scoring using TargetBuilder pipeline."""
            final_weights = {**self.default_weights, **(request.weights or {})}

            # Score targets in parallel using TargetBuilder
            tasks = [
                self.score_single_target(
                    request.disease,
                    target,
                    final_weights,
                    targets_context=request.targets
                )
                for target in request.targets
            ]

            target_scores = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            valid_scores = []
            for i, result in enumerate(target_scores):
                if isinstance(result, Exception):
                    logger.error(f"Target scoring failed for {request.targets[i]}: {result}")
                    # Create error TargetScore
                    error_score = TargetScore(
                        target=request.targets[i],
                        total_score=0.1,
                        breakdown=TargetBreakdown(),
                        evidence_refs=[f"Error: {str(result)[:100]}"],
                        data_version="Error",
                        explanation={"target": request.targets[i], "error": str(result)},
                        timestamp=get_utc_now(),
                        warnings=[f"Scoring failed: {str(result)}"]
                    )
                    valid_scores.append(error_score)
                else:
                    valid_scores.append(result)

            # Build metadata
            metadata = {
                "data_version": "TargetBuilder-v1.0.0-phase1c",
                "meta": {
                    "cached": False,  # TODO: Extract from lineage
                    "fetch_ms": 0.0,  # TODO: Extract from lineage
                    "cache_hit_rate": 0.0,
                    "total_calls": len(request.targets)
                },
                "system_info": {
                    "pipeline": "TargetBuilder",
                    "targets_processed": len(valid_scores),
                    "successful_scores": sum(1 for ts in valid_scores if ts.total_score > 0.1)
                }
            }

            return valid_scores, metadata

        def validate_request(self, request: ScoreRequest) -> Tuple[bool, str]:
            """Validate scoring request."""
            if not request.targets:
                return False, "No targets provided"
            if len(request.targets) > 50:
                return False, "Too many targets (max 50)"
            if not request.disease:
                return False, "Disease identifier required"

            # Validate weights
            for channel, weight in (request.weights or {}).items():
                if not isinstance(weight, (int, float)):
                    return False, f"Weight for {channel} must be numeric"
                if not 0 <= float(weight) <= 1:
                    return False, f"Weight for {channel} must be between 0 and 1"

            weight_sum = sum(float(w) for w in (request.weights or {}).values())
            if weight_sum > 1.2 or weight_sum < 0.8:
                return False, f"Weights should sum to ~1.0 (current sum: {weight_sum:.2f})"

            return True, ""

    # ========================
    # Simulation and analysis functions (keeping for API compatibility)
    # ========================

    def simulate_weight_perturbations(
            target_scores: List[TargetScore],
            base_weights: Dict[str, float],
            n_samples: int = 200,
            dirichlet_alpha: float = 80.0
    ) -> Dict:
        """Weight perturbation simulation (keeping existing implementation)."""
        if not target_scores:
            return {
                "stability": {},
                "kendall_tau_mean": 0.0,
                "samples": 0,
                "weight_stats": {}
            }

        try:
            channels = ["genetics", "ppi", "pathway", "safety", "modality_fit"]
            base_weight_vector = np.array([base_weights.get(ch, 0.0) for ch in channels])
            alpha_vector = base_weight_vector * dirichlet_alpha
            alpha_vector = np.maximum(alpha_vector, 0.1)
            sampled_weights = dirichlet.rvs(alpha_vector, size=n_samples)

            # Extract channel scores
            target_channel_scores = []
            target_names = []

            for ts in target_scores:
                target_names.append(ts.target)
                breakdown = ts.breakdown

                scores = {
                    "genetics": float(breakdown.genetics or 0.0),
                    "ppi": float(breakdown.ppi_proximity or 0.0),
                    "pathway": float(breakdown.pathway_enrichment or 0.0),
                    "safety": float(breakdown.safety_off_tissue or 0.0),
                    "modality_fit": 0.0
                }

                if breakdown.modality_fit:
                    if isinstance(breakdown.modality_fit, dict):
                        scores["modality_fit"] = float(breakdown.modality_fit.get("overall_druggability", 0.0))
                    else:
                        scores["modality_fit"] = float(getattr(breakdown.modality_fit, "overall_druggability", 0.0))

                score_vector = np.array([scores[ch] for ch in channels])
                target_channel_scores.append(score_vector)

            target_channel_scores = np.array(target_channel_scores)

            # Simulate rankings
            rank_matrices = np.zeros((len(target_names), n_samples), dtype=int)

            for sample_idx in range(n_samples):
                weight_sample = sampled_weights[sample_idx]
                sample_scores = []

                for target_idx in range(len(target_names)):
                    target_scores_vec = target_channel_scores[target_idx]
                    adjusted_scores = target_scores_vec.copy()
                    safety_idx = channels.index("safety")
                    adjusted_scores[safety_idx] = 1.0 - adjusted_scores[safety_idx]
                    weighted_score = np.dot(adjusted_scores, weight_sample)
                    sample_scores.append(weighted_score)

                sample_ranks = len(sample_scores) + 1 - np.argsort(np.argsort(sample_scores))
                rank_matrices[:, sample_idx] = sample_ranks

            # Compute stability metrics
            stability_results = {}
            for target_idx, target_name in enumerate(target_names):
                target_ranks = rank_matrices[target_idx, :]
                unique_ranks, counts = np.unique(target_ranks, return_counts=True)
                histogram = {int(rank): int(count) for rank, count in zip(unique_ranks, counts)}
                mode_rank = int(unique_ranks[np.argmax(counts)])

                probabilities = counts / n_samples
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
                max_entropy = np.log2(len(target_names))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

                stability_results[target_name] = {
                    "mode_rank": mode_rank,
                    "entropy": float(normalized_entropy),
                    "histogram": histogram,
                    "rank_range": [int(np.min(target_ranks)), int(np.max(target_ranks))],
                    "rank_std": float(np.std(target_ranks))
                }

            # Kendall tau
            kendall_taus = []
            base_ranking = rank_matrices[:, 0]
            for sample_idx in range(1, min(n_samples, 50)):
                sample_ranking = rank_matrices[:, sample_idx]
                try:
                    tau, _ = kendalltau(base_ranking, sample_ranking)
                    if not np.isnan(tau):
                        kendall_taus.append(tau)
                except Exception:
                    continue

            kendall_tau_mean = float(np.mean(kendall_taus)) if kendall_taus else 0.0

            # Weight stats
            weight_stats = {}
            for ch_idx, channel in enumerate(channels):
                channel_weights = sampled_weights[:, ch_idx]
                weight_stats[channel] = {
                    "mean": float(np.mean(channel_weights)),
                    "std": float(np.std(channel_weights)),
                    "min": float(np.min(channel_weights)),
                    "max": float(np.max(channel_weights)),
                    "base": float(base_weight_vector[ch_idx])
                }

            return {
                "stability": stability_results,
                "kendall_tau_mean": kendall_tau_mean,
                "samples": n_samples,
                "weight_stats": weight_stats
            }

        except Exception as e:
            logger.error(f"Weight perturbation simulation failed: {e}")
            return {"stability": {}, "kendall_tau_mean": 0.0, "samples": 0, "error": str(e)}

    def compute_channel_ablation(target_scores: List[TargetScore], weights: Dict[str, float]) -> List[Dict]:
        """Channel ablation analysis (keeping existing implementation)."""
        if not target_scores or not weights:
            return []

        try:
            channels = ["genetics", "ppi", "pathway", "safety", "modality_fit"]

            # Baseline rankings
            baseline_scores = {}
            baseline_rankings = {}
            sorted_targets = sorted(target_scores, key=lambda x: x.total_score, reverse=True)

            for i, ts in enumerate(sorted_targets):
                baseline_scores[ts.target] = ts.total_score
                baseline_rankings[ts.target] = i + 1

            ablation_results = []

            for ablated_channel in channels:
                ablated_weights = weights.copy()
                ablated_weights[ablated_channel] = 0.0

                remaining_weight = sum(w for ch, w in ablated_weights.items() if w > 0)
                if remaining_weight > 0:
                    for ch in ablated_weights:
                        if ablated_weights[ch] > 0:
                            ablated_weights[ch] = ablated_weights[ch] / remaining_weight
                else:
                    continue

                # Recompute scores
                channel_deltas = []
                for ts in target_scores:
                    breakdown = ts.breakdown

                    channel_scores = {
                        "genetics": float(breakdown.genetics or 0.0),
                        "ppi": float(breakdown.ppi_proximity or 0.0),
                        "pathway": float(breakdown.pathway_enrichment or 0.0),
                        "safety": float(breakdown.safety_off_tissue or 0.0),
                        "modality_fit": 0.0
                    }

                    if breakdown.modality_fit:
                        if isinstance(breakdown.modality_fit, dict):
                            channel_scores["modality_fit"] = float(
                                breakdown.modality_fit.get("overall_druggability", 0.0))
                        else:
                            channel_scores["modality_fit"] = float(
                                getattr(breakdown.modality_fit, "overall_druggability", 0.0))

                    ablated_score = 0.0
                    for channel, weight in ablated_weights.items():
                        if weight > 0 and channel in channel_scores:
                            score = channel_scores[channel]
                            if channel == "safety":
                                score = 1.0 - score
                            score = max(0.0, min(1.0, score))
                            ablated_score += score * weight

                    channel_deltas.append({
                        "target": ts.target,
                        "score_drop": float(baseline_scores[ts.target] - ablated_score),
                        "original_score": float(baseline_scores[ts.target]),
                        "ablated_score": float(ablated_score)
                    })

                avg_drop = sum(d["score_drop"] for d in channel_deltas) / len(channel_deltas)
                max_drop = max(d["score_drop"] for d in channel_deltas) if channel_deltas else 0.0

                ablation_results.append({
                    "channel": ablated_channel,
                    "avg_score_drop": float(avg_drop),
                    "max_score_drop": float(max_drop),
                    "targets_affected": int(sum(1 for d in channel_deltas if d["score_drop"] > 0.01)),
                    "delta": channel_deltas
                })

            ablation_results.sort(key=lambda x: x["avg_score_drop"], reverse=True)
            return ablation_results

        except Exception as e:
            logger.error(f"Channel ablation analysis failed: {e}")
            return []

    # ========================
    # Global instances for API compatibility
    # ========================

    target_scorer = LegacyTargetScorer()

    async def score_targets(request: ScoreRequest) -> Tuple[List[TargetScore], Dict]:
        """Main scoring function for API endpoints (legacy compatibility)."""
        return await target_scorer.score_targets_batch(request)

    def validate_score_request(request: ScoreRequest) -> Tuple[bool, str]:
        """Validate scoring request (legacy compatibility)."""
        return target_scorer.validate_request(request)
    async def _fetch_pathway_data(self) -> Dict[str, Any]:
        """Fetch pathway enrichment data."""
        try:
            targets_context = [self.gene]  # Simplified
            pathway_score, pathway_refs = await compute_pathway_enrichment(self.gene, targets_context)

            return {
                "score": pathway_score,
                "evidence_refs": pathway_refs,
                "status": "ok" if pathway_score > 0 else "data_missing"
            }
        except Exception as e:
            logger.error(f"Pathway data fetch failed for {self.gene}: {e}")
            return {"status": "error", "error": str(e)}

    async def _fetch_safety_data(self) -> Dict[str, Any]:
        try:
            disease_id, note = self._normalize_disease()
            ot_data = await fetch_ot_association(disease_id, self.gene)
            safety_score, safety_refs = await compute_safety_penalty(self.gene, ot_data)
            status = "ok" if safety_score is not None else "data_missing"
            out = {"score": safety_score, "evidence_refs": safety_refs, "status": status}
            if note: out["quality_note"] = note
            return out
        except Exception as e:
            logger.error(f"Safety data fetch failed for {self.gene}: {e}")
            return {"status": "error", "error": str(e)}

    async def _fetch_modality_data(self) -> Dict[str, Any]:
        """Fetch modality fit data."""
        try:
            modality_scores, modality_refs = await compute_modality_fit(self.gene, None)

            return {
                "scores": modality_scores,
                "evidence_refs": modality_refs,
                "status": "ok" if modality_scores else "data_missing"
            }
        except Exception as e:
            logger.error(f"Modality data fetch failed for {self.gene}: {e}")
            return {"status": "error", "error": str(e)}

    async def validate(self, raw_data: Dict[str, Any]) -> Dict[str, ChannelScore]:
        """
        Validate raw channel data using DataQualityValidator.

        Args:
            raw_data: Raw data from fetch_all()

        Returns:
            Dict of validated ChannelScore objects
        """
        validated = {}
        validation_start = time.time()

        for channel, data in raw_data.items():
            try:
                if channel == "genetics" and isinstance(data, ChannelScore):
                    # Genetics already returns ChannelScore
                    validated[channel] = data
                    continue

                # Convert other channels to ChannelScore format
                if isinstance(data, dict) and data.get("status") == "error":
                    validated[channel] = ChannelScore(
                        name=channel,
                        score=None,
                        status="error",
                        components={},
                        evidence=[],
                        quality=DataQualityFlags(notes=f"Fetch error: {data.get('error', '')[:100]}")
                    )
                elif isinstance(data, dict) and data.get("status") == "data_missing":
                    validated[channel] = ChannelScore(
                        name=channel,
                        score=None,
                        status="data_missing",
                        components={},
                        evidence=[],
                        quality=DataQualityFlags(partial=True, notes="No data available")
                    )
                else:
                    # Validate and convert to ChannelScore
                    validation_result = self.validator.validate(channel, data)

                    score = None
                    evidence_refs = []
                    components = {}

                    if isinstance(data, dict):
                        score = data.get("score")
                        evidence_refs = self._convert_evidence_refs(data.get("evidence_refs", []))
                        components = data.get("components", {})

                        # Handle modality scores specially
                        if channel == "modality_fit" and "scores" in data:
                            modality_scores = data["scores"]
                            if isinstance(modality_scores, dict):
                                components.update(modality_scores)
                                score = modality_scores.get("overall_druggability", 0.0)
                            elif hasattr(modality_scores, "overall_druggability"):
                                score = getattr(modality_scores, "overall_druggability", 0.0)
                    incoming_status = (data.get("status") if isinstance(data, dict) else None)
                    if incoming_status in {"ok", "data_missing", "error"}:
                        status = incoming_status
                    else:
                        status = "ok" if (score is not None) else "data_missing"

                    validated[channel] = ChannelScore(
                        name=channel,
                        score=score,
                        status= status,
                        components=components,
                        evidence=evidence_refs,
                        quality=validation_result.quality
                    )

            except Exception as e:
                logger.error(f"Validation failed for {channel}: {e}")
                validated[channel] = ChannelScore(
                    name=channel,
                    score=None,
                    status="error",
                    components={"validation_error": str(e)[:100]},
                    evidence=[],
                    quality=DataQualityFlags(notes=f"Validation error: {str(e)[:100]}")
                )

        validation_time = (time.time() - validation_start) * 1000
        self.lineage["transforms"].append(f"validate: {validation_time:.1f}ms")

        return validated

    def _convert_evidence_refs(self, evidence_strings: List[str]) -> List[EvidenceRef]:
        """Convert legacy evidence strings like 'Source:stringdb' into structured EvidenceRef."""
        out: List[EvidenceRef] = []
        for s in evidence_strings:
            try:
                st = s.strip()
                if st.startswith("PMID:"):
                    pmid = st.split("PMID:")[1].split()[0]
                    out.append(EvidenceRef(source="pubmed",
                                           pmid=pmid,
                                           url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                           timestamp=get_utc_now()))
                    continue
                if st.lower().startswith("source:"):
                    src = st.split(":", 1)[1].strip().lower()
                    if src in {"stringdb", "string"}:
                        out.append(EvidenceRef(source="stringdb",
                                               title="STRING evidence",
                                               url="https://string-db.org/",
                                               timestamp=get_utc_now()))
                    elif src in {"opentargets", "ot"}:
                        out.append(EvidenceRef(source="opentargets",
                                               title="OpenTargets evidence",
                                               url="https://platform.opentargets.org/",
                                               timestamp=get_utc_now()))
                    elif src in {"fallback"}:
                        out.append(EvidenceRef(source="fallback",
                                               title="Fallback evidence",
                                               timestamp=get_utc_now()))
                    else:
                        out.append(EvidenceRef(source=src or "unknown",
                                               title=st[:100],
                                               timestamp=get_utc_now()))
                    continue
                if "STRING:" in st:
                    out.append(EvidenceRef(source="stringdb",
                                           title=st[:100],
                                           url="https://string-db.org/",
                                           timestamp=get_utc_now()))
                elif "OpenTargets:" in st or "OT-" in st:
                    out.append(EvidenceRef(source="opentargets",
                                           title=st[:100],
                                           url="https://platform.opentargets.org/",
                                           timestamp=get_utc_now()))
                else:
                    out.append(EvidenceRef(source="unknown",
                                           title=st[:100],
                                           timestamp=get_utc_now()))
            except Exception as e:
                logger.warning(f"Evidence parse error: {s} ({e})")
        return out

    async def compute_scores(self, validated_data: Dict[str, ChannelScore], weights: Dict[str, float]) -> Dict[
        str, float]:
        """
        Combine channel scores using config-driven weights.

        Args:
            validated_data: Validated channel data
            weights: Channel weights configuration

        Returns:
            Dict with final scores and metadata
        """
        compute_start = time.time()

        # Extract scores from validated channels
        channel_scores = {}
        available_channels = []
        total_weight = 0.0

        for channel, channel_data in validated_data.items():
            if channel_data.status == "ok" and channel_data.score is not None:
                score = float(channel_data.score)

                # Handle safety inversion (safety is penalty)
                if channel == "safety":
                    score = 1.0 - score

                # Ensure bounds
                score = max(0.0, min(1.0, score))
                channel_scores[channel] = score
                available_channels.append(channel)

                if channel in weights:
                    total_weight += weights[channel]

        # Normalize weights for available channels
        if total_weight > 0:
            normalized_weights = {
                ch: weights.get(ch, 0.0) / total_weight
                for ch in available_channels
            }
        else:
            # All channels failed - use equal weights
            normalized_weights = {ch: 1.0 / len(available_channels) for ch in available_channels}

        # Compute weighted total score
        combined_score = 0.0
        for channel, score in channel_scores.items():
            weight = normalized_weights.get(channel, 0.0)
            combined_score += score * weight

        # Ensure minimum score
        combined_score = max(0.1, min(1.0, combined_score))

        compute_time = (time.time() - compute_start) * 1000
        self.lineage["transforms"].append(f"compute_scores: {compute_time:.1f}ms")

        logger.info(
            f"Scores computed for {self.gene}",
            extra={
                "gene": self.gene,
                "combined_score": combined_score,
                "available_channels": available_channels,
                "total_weight": total_weight
            }
        )

        return {
            "combined_score": combined_score,
            "channel_scores": channel_scores,
            "normalized_weights": normalized_weights,
            "available_channels": available_channels
        }

    async def assemble(self, weights: Dict[str, float]) -> TargetScoreBundle:
        """
        Assemble complete TargetScoreBundle with lineage tracking.

        Args:
            weights: Channel weights configuration

        Returns:
            Complete TargetScoreBundle
        """
        # Execute pipeline stages
        raw_data = await self.fetch_all()
        validated_data = await self.validate(raw_data)
        score_results = await self.compute_scores(validated_data, weights)

        # Build final bundle
        bundle = TargetScoreBundle(
            gene=self.gene,
            disease=self.disease,
            channels=validated_data,
            combined_score=score_results["combined_score"],
            lineage=self.lineage,
            timestamp=self.timestamp
        )

        # Update lineage with final results
        self.lineage["final_score"] = score_results["combined_score"]
        self.lineage["available_channels"] = score_results["available_channels"]
        self.lineage["pipeline_completed"] = get_utc_now().isoformat()

        return bundle




# ========================
# Legacy compatibility layer for existing API
# ========================

class LegacyTargetScorer:
    """Legacy scorer for backward compatibility with existing main.py."""

    def __init__(self):
        self.default_weights = DEFAULT_WEIGHTS

    async def score_single_target(
            self,
            disease: str,
            target: str,
            weights: Dict[str, float],
            targets_context: List[str] = None,
            version_manager=None
    ) -> TargetScore:
        """Legacy single target scoring with TargetBuilder pipeline."""

        # Use new TargetBuilder pipeline
        builder = TargetBuilder(target, disease)
        bundle = await builder.assemble(weights)
        channels_export = {name: cs.model_dump() for name, cs in bundle.channels.items()}

        # Convert TargetScoreBundle to legacy TargetScore format
        breakdown = TargetBreakdown()
        all_evidence_refs = []

        # Extract scores from channels
        for channel_name, channel_score in bundle.channels.items():
            if channel_name == "genetics":
                breakdown.genetics = channel_score.score
            elif channel_name == "ppi":
                breakdown.ppi_proximity = channel_score.score
            elif channel_name == "pathway":
                breakdown.pathway_enrichment = channel_score.score
            elif channel_name == "safety":
                breakdown.safety_off_tissue = channel_score.score
            elif channel_name == "modality_fit":
                breakdown.modality_fit = channel_score.components

            # Collect evidence references
            for evidence in channel_score.evidence:
                if evidence.pmid:
                    all_evidence_refs.append(f"PMID:{evidence.pmid}")
                all_evidence_refs.append(f"Source:{evidence.source}")

        # Build explanation (simplified for compatibility)
        explanation = {
            "target": target,
            "total_weighted_score": bundle.combined_score or 0.0,
            "confidence_level": "medium",
            "key_insights": [f"Scored using TargetBuilder pipeline v{bundle.lineage.get('pipeline_version', '1.0')}"]
        }

        # Data version from lineage
        data_version = f"Pipeline-{bundle.lineage.get('pipeline_version', 'unknown')}"

        return TargetScore(
            target=target,
            total_score=bundle.combined_score or 0.0,
            breakdown=breakdown,
            evidence_refs=list(dict.fromkeys(all_evidence_refs)),  # Remove duplicates
            data_version=data_version,
            explanation=explanation,
            timestamp=bundle.timestamp,
            warnings=None,
            channels=bundle.channels,
        )

    async def score_targets_batch(self, request: ScoreRequest) -> Tuple[List[TargetScore], Dict]:
        """Legacy batch scoring using TargetBuilder pipeline."""
        final_weights = {**self.default_weights, **(request.weights or {})}

        # Score targets in parallel using TargetBuilder
        tasks = [
            self.score_single_target(
                request.disease,
                target,
                final_weights,
                targets_context=request.targets
            )
            for target in request.targets
        ]

        target_scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        valid_scores = []
        for i, result in enumerate(target_scores):
            if isinstance(result, Exception):
                logger.error(f"Target scoring failed for {request.targets[i]}: {result}")
                # Create error TargetScore
                error_score = TargetScore(
                    target=request.targets[i],
                    total_score=0.1,
                    breakdown=TargetBreakdown(),
                    evidence_refs=[f"Error: {str(result)[:100]}"],
                    data_version="Error",
                    explanation={"target": request.targets[i], "error": str(result)},
                    timestamp=get_utc_now(),
                    warnings=[f"Scoring failed: {str(result)}"]
                )
                valid_scores.append(error_score)
            else:
                valid_scores.append(result)

        # Build metadata
        metadata = {
            "data_version": "TargetBuilder-v1.0.0-phase1c",
            "meta": {
                "cached": False,  # TODO: Extract from lineage
                "fetch_ms": 0.0,  # TODO: Extract from lineage
                "cache_hit_rate": 0.0,
                "total_calls": len(request.targets)
            },
            "system_info": {
                "pipeline": "TargetBuilder",
                "targets_processed": len(valid_scores),
                "successful_scores": sum(1 for ts in valid_scores if ts.total_score > 0.1)
            }
        }

        return valid_scores, metadata

    def validate_request(self, request: ScoreRequest) -> Tuple[bool, str]:
        """Validate scoring request."""
        if not request.targets:
            return False, "No targets provided"
        if len(request.targets) > 50:
            return False, "Too many targets (max 50)"
        if not request.disease:
            return False, "Disease identifier required"

        # Validate weights
        for channel, weight in (request.weights or {}).items():
            if not isinstance(weight, (int, float)):
                return False, f"Weight for {channel} must be numeric"
            if not 0 <= float(weight) <= 1:
                return False, f"Weight for {channel} must be between 0 and 1"

        weight_sum = sum(float(w) for w in (request.weights or {}).values())
        if weight_sum > 1.2 or weight_sum < 0.8:
            return False, f"Weights should sum to ~1.0 (current sum: {weight_sum:.2f})"

        return True, ""


# ========================
# Simulation and analysis functions (keeping for API compatibility)
# ========================

def simulate_weight_perturbations(
        target_scores: List[TargetScore],
        base_weights: Dict[str, float],
        n_samples: int = 200,
        dirichlet_alpha: float = 80.0
) -> Dict:
    """Weight perturbation simulation (keeping existing implementation)."""
    if not target_scores:
        return {
            "stability": {},
            "kendall_tau_mean": 0.0,
            "samples": 0,
            "weight_stats": {}
        }

    try:
        channels = ["genetics", "ppi", "pathway", "safety", "modality_fit"]
        base_weight_vector = np.array([base_weights.get(ch, 0.0) for ch in channels])
        alpha_vector = base_weight_vector * dirichlet_alpha
        alpha_vector = np.maximum(alpha_vector, 0.1)
        sampled_weights = dirichlet.rvs(alpha_vector, size=n_samples)

        # Extract channel scores
        target_channel_scores = []
        target_names = []

        for ts in target_scores:
            target_names.append(ts.target)
            breakdown = ts.breakdown

            scores = {
                "genetics": float(breakdown.genetics or 0.0),
                "ppi": float(breakdown.ppi_proximity or 0.0),
                "pathway": float(breakdown.pathway_enrichment or 0.0),
                "safety": float(breakdown.safety_off_tissue or 0.0),
                "modality_fit": 0.0
            }

            if breakdown.modality_fit:
                if isinstance(breakdown.modality_fit, dict):
                    scores["modality_fit"] = float(breakdown.modality_fit.get("overall_druggability", 0.0))
                else:
                    scores["modality_fit"] = float(getattr(breakdown.modality_fit, "overall_druggability", 0.0))

            score_vector = np.array([scores[ch] for ch in channels])
            target_channel_scores.append(score_vector)

        target_channel_scores = np.array(target_channel_scores)

        # Simulate rankings
        rank_matrices = np.zeros((len(target_names), n_samples), dtype=int)

        for sample_idx in range(n_samples):
            weight_sample = sampled_weights[sample_idx]
            sample_scores = []

            for target_idx in range(len(target_names)):
                target_scores_vec = target_channel_scores[target_idx]
                adjusted_scores = target_scores_vec.copy()
                safety_idx = channels.index("safety")
                adjusted_scores[safety_idx] = 1.0 - adjusted_scores[safety_idx]
                weighted_score = np.dot(adjusted_scores, weight_sample)
                sample_scores.append(weighted_score)

            sample_ranks = len(sample_scores) + 1 - np.argsort(np.argsort(sample_scores))
            rank_matrices[:, sample_idx] = sample_ranks

        # Compute stability metrics
        stability_results = {}
        for target_idx, target_name in enumerate(target_names):
            target_ranks = rank_matrices[target_idx, :]
            unique_ranks, counts = np.unique(target_ranks, return_counts=True)
            histogram = {int(rank): int(count) for rank, count in zip(unique_ranks, counts)}
            mode_rank = int(unique_ranks[np.argmax(counts)])

            probabilities = counts / n_samples
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
            max_entropy = np.log2(len(target_names))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            stability_results[target_name] = {
                "mode_rank": mode_rank,
                "entropy": float(normalized_entropy),
                "histogram": histogram,
                "rank_range": [int(np.min(target_ranks)), int(np.max(target_ranks))],
                "rank_std": float(np.std(target_ranks))
            }

        # Kendall tau
        kendall_taus = []
        base_ranking = rank_matrices[:, 0]
        for sample_idx in range(1, min(n_samples, 50)):
            sample_ranking = rank_matrices[:, sample_idx]
            try:
                tau, _ = kendalltau(base_ranking, sample_ranking)
                if not np.isnan(tau):
                    kendall_taus.append(tau)
            except Exception:
                continue

        kendall_tau_mean = float(np.mean(kendall_taus)) if kendall_taus else 0.0

        # Weight stats
        weight_stats = {}
        for ch_idx, channel in enumerate(channels):
            channel_weights = sampled_weights[:, ch_idx]
            weight_stats[channel] = {
                "mean": float(np.mean(channel_weights)),
                "std": float(np.std(channel_weights)),
                "min": float(np.min(channel_weights)),
                "max": float(np.max(channel_weights)),
                "base": float(base_weight_vector[ch_idx])
            }

        return {
            "stability": stability_results,
            "kendall_tau_mean": kendall_tau_mean,
            "samples": n_samples,
            "weight_stats": weight_stats
        }

    except Exception as e:
        logger.error(f"Weight perturbation simulation failed: {e}")
        return {"stability": {}, "kendall_tau_mean": 0.0, "samples": 0, "error": str(e)}


def compute_channel_ablation(target_scores: List[TargetScore], weights: Dict[str, float]) -> List[Dict]:
    """Channel ablation analysis (keeping existing implementation)."""
    if not target_scores or not weights:
        return []

    try:
        channels = ["genetics", "ppi", "pathway", "safety", "modality_fit"]

        # Baseline rankings
        baseline_scores = {}
        baseline_rankings = {}
        sorted_targets = sorted(target_scores, key=lambda x: x.total_score, reverse=True)

        for i, ts in enumerate(sorted_targets):
            baseline_scores[ts.target] = ts.total_score
            baseline_rankings[ts.target] = i + 1

        ablation_results = []

        for ablated_channel in channels:
            ablated_weights = weights.copy()
            ablated_weights[ablated_channel] = 0.0

            remaining_weight = sum(w for ch, w in ablated_weights.items() if w > 0)
            if remaining_weight > 0:
                for ch in ablated_weights:
                    if ablated_weights[ch] > 0:
                        ablated_weights[ch] = ablated_weights[ch] / remaining_weight
            else:
                continue

            # Recompute scores
            channel_deltas = []
            for ts in target_scores:
                breakdown = ts.breakdown

                channel_scores = {
                    "genetics": float(breakdown.genetics or 0.0),
                    "ppi": float(breakdown.ppi_proximity or 0.0),
                    "pathway": float(breakdown.pathway_enrichment or 0.0),
                    "safety": float(breakdown.safety_off_tissue or 0.0),
                    "modality_fit": 0.0
                }

                if breakdown.modality_fit:
                    if isinstance(breakdown.modality_fit, dict):
                        channel_scores["modality_fit"] = float(breakdown.modality_fit.get("overall_druggability", 0.0))
                    else:
                        channel_scores["modality_fit"] = float(
                            getattr(breakdown.modality_fit, "overall_druggability", 0.0))

                ablated_score = 0.0
                for channel, weight in ablated_weights.items():
                    if weight > 0 and channel in channel_scores:
                        score = channel_scores[channel]
                        if channel == "safety":
                            score = 1.0 - score
                        score = max(0.0, min(1.0, score))
                        ablated_score += score * weight

                channel_deltas.append({
                    "target": ts.target,
                    "score_drop": float(baseline_scores[ts.target] - ablated_score),
                    "original_score": float(baseline_scores[ts.target]),
                    "ablated_score": float(ablated_score)
                })

            avg_drop = sum(d["score_drop"] for d in channel_deltas) / len(channel_deltas)
            max_drop = max(d["score_drop"] for d in channel_deltas) if channel_deltas else 0.0

            ablation_results.append({
                "channel": ablated_channel,
                "avg_score_drop": float(avg_drop),
                "max_score_drop": float(max_drop),
                "targets_affected": int(sum(1 for d in channel_deltas if d["score_drop"] > 0.01)),
                "delta": channel_deltas
            })

        ablation_results.sort(key=lambda x: x["avg_score_drop"], reverse=True)
        return ablation_results

    except Exception as e:
        logger.error(f"Channel ablation analysis failed: {e}")
        return []


# ========================
# Global instances for API compatibility
# ========================

target_scorer = LegacyTargetScorer()


async def score_targets(request: ScoreRequest) -> Tuple[List[TargetScore], Dict]:
    """Main scoring function for API endpoints (legacy compatibility)."""
    return await target_scorer.score_targets_batch(request)


def validate_score_request(request: ScoreRequest) -> Tuple[bool, str]:
    """Validate scoring request (legacy compatibility)."""
    return target_scorer.validate_request(request)