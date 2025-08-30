# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Pydantic schemas for VantAI Target Scoreboard API - Phase 1.
Kanonik modeller: gerçek veri entegrasyonu için production-ready şemalar.
"""
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Dict, List, Any, Union, Literal
from datetime import datetime
import re


def schema_version() -> str:
    """Return repo-scoped schema version."""
    return "v1.0.0-phase1"


def get_utc_now() -> datetime:
    """Consistent UTC timestamp helper."""
    return datetime.utcnow()


# ========================
# Core Data Quality Models
# ========================

class DataQualityFlags(BaseModel):
    """Data quality indicators for evidence tracking."""
    stale: bool = Field(False, description="Data older than acceptable threshold")
    partial: bool = Field(False, description="Incomplete data returned")
    schema_version: str = Field(default_factory=schema_version, description="Schema version used")
    notes: Optional[str] = Field(None, description="Additional quality notes")


class ValidationResult(BaseModel):
    """Result of data validation process."""
    ok: bool = Field(..., description="Whether validation passed")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    quality: DataQualityFlags = Field(default_factory=DataQualityFlags, description="Quality flags")


# ========================
# Evidence and Reference Models
# ========================

class EvidenceRef(BaseModel):
    """Evidence reference with external links and quality metadata."""
    source: str = Field(..., description="Data source identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID if literature evidence")
    title: Optional[str] = Field(None, description="Article or evidence title")
    journal: Optional[str] = Field(None, description="Journal name for literature")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Publication year")
    url: Optional[str] = Field(None, description="Direct URL to evidence")
    source_quality: Optional[str] = Field(None, description="Source quality assessment")
    timestamp: datetime = Field(default_factory=get_utc_now, description="When evidence was retrieved")


# ========================
# Biological Data Models
# ========================

class AssociationRecord(BaseModel):
    """Gene-disease association from external databases."""
    gene: str = Field(..., description="Gene symbol")
    disease: str = Field(..., description="Disease identifier")
    score: float = Field(..., ge=0, le=1, description="Association strength score")
    pval: Optional[float] = Field(None, ge=0, le=1, description="Statistical p-value if available")
    source: str = Field(..., description="Source database")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Retrieval timestamp")
    evidence: List[EvidenceRef] = Field(default_factory=list, description="Supporting evidence")


class PPIEdge(BaseModel):
    """Protein-protein interaction edge."""
    source_gene: str = Field(..., description="Source gene symbol")
    partner: str = Field(..., description="Interaction partner gene symbol")
    confidence: float = Field(..., ge=0, le=1, description="Interaction confidence score")
    evidence: List[EvidenceRef] = Field(default_factory=list, description="Supporting evidence")


class PPINetwork(BaseModel):
    """Complete protein-protein interaction network for a gene."""
    gene: str = Field(..., description="Central gene symbol")
    edges: List[PPIEdge] = Field(default_factory=list, description="Network edges")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Network retrieval time")
    source: str = Field(..., description="Network data source")


class ExpressionRecord(BaseModel):
    """Gene expression measurement."""
    gene: str = Field(..., description="Gene symbol")
    tissue: str = Field(..., description="Tissue type")
    value: float = Field(..., description="Expression value")
    unit: Optional[str] = Field(None, description="Expression unit (TPM, FPKM, etc.)")
    quantile: Optional[float] = Field(None, ge=0, le=1, description="Expression quantile in tissue")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Measurement timestamp")
    source: str = Field(..., description="Expression data source")


class StructureConfidence(BaseModel):
    """Protein structure confidence scores."""
    gene: str = Field(..., description="Gene symbol")
    plddt_mean: Optional[float] = Field(None, ge=0, le=100, description="Mean pLDDT confidence")
    pae_mean: Optional[float] = Field(None, ge=0, description="Mean PAE (predicted aligned error)")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Structure data timestamp")
    source: str = Field(..., description="Structure database source")


# ========================
# Channel Scoring Models
# ========================

class ChannelScore(BaseModel):
    """Individual channel contribution to target score."""
    name: str = Field(..., description="Channel identifier")
    score: Optional[float] = Field(None, ge=0, le=1, description="Channel score (None if data missing)")
    status: Literal["ok", "data_missing", "error"] = Field(..., description="Channel status")
    components: Dict[str, float] = Field(default_factory=dict, description="Sub-component scores")
    evidence: List[EvidenceRef] = Field(default_factory=list, description="Supporting evidence")
    quality: DataQualityFlags = Field(default_factory=DataQualityFlags, description="Data quality flags")



class TargetScoreBundle(BaseModel):
    """Complete scoring bundle for a target-disease pair."""
    gene: str = Field(..., description="Target gene symbol")
    disease: Optional[str] = Field(None, description="Disease identifier")
    channels: Dict[str, ChannelScore] = Field(default_factory=dict, description="Channel scores")
    combined_score: Optional[float] = Field(None, ge=0, le=1, description="Weighted combined score")
    lineage: Dict[str, Any] = Field(default_factory=dict, description="Data provenance tracking")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Scoring timestamp")


# ========================
# API Request/Response Models (keeping existing for compatibility)
# ========================

class ScoreRequest(BaseModel):
    """Request model for target scoring endpoint."""
    disease: str = Field(..., description="Disease identifier (EFO, MONDO, etc.)")
    targets: List[str] = Field(..., min_length=1, max_length=50, description="List of target gene symbols")
    weights: Dict[str, float] = Field(
        default={
            "genetics": 0.35,
            "ppi": 0.25,
            "pathway": 0.20,
            "safety": 0.10,
            "modality_fit": 0.10
        },
        description="Channel weights for scoring (must sum to ~1.0)"
    )

    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Validate that weights are reasonable."""
        if not isinstance(v, dict):
            raise ValueError("Weights must be a dictionary")

        # Check individual weight bounds
        for channel, weight in v.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for {channel} must be between 0 and 1")

        # Check total weight sum
        total_weight = sum(v.values())
        if not 0.8 <= total_weight <= 1.2:
            raise ValueError(f"Total weight sum {total_weight:.2f} should be approximately 1.0")

        return v

    @field_validator('targets')
    @classmethod
    def validate_targets(cls, v):
        """Validate target gene symbols."""
        if not v:
            raise ValueError("At least one target must be provided")

        # Clean and validate gene symbols
        cleaned_targets = []
        for target in v:
            if not isinstance(target, str):
                raise ValueError("All targets must be strings")

            cleaned = target.strip().upper()
            if len(cleaned) < 2:
                raise ValueError(f"Target '{target}' is too short")

            if not cleaned.replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"Target '{target}' contains invalid characters")

            cleaned_targets.append(cleaned)

        return cleaned_targets


class ModalityFitScores(BaseModel):
    """Detailed modality fit subscores."""
    e3_coexpr: float = Field(..., ge=0, le=1, description="E3 ligase co-expression score")
    ternary_proxy: float = Field(..., ge=0, le=1, description="Ternary complex formation proxy")
    ppi_hotspot: float = Field(..., ge=0, le=1, description="PPI hotspot druggability score")
    overall_druggability: float = Field(..., ge=0, le=1, description="Combined modality fitness")
    protac_degrader: float = Field(..., ge=0, le=1, description="PROTAC/degrader suitability")
    small_molecule: float = Field(..., ge=0, le=1, description="Small molecule druggability")
    molecular_glue: Optional[float] = Field(None, ge=0, le=1, description="Molecular glue potential")
    adc_score: Optional[float] = Field(None, ge=0, le=1, description="ADC targeting potential")


class TargetBreakdown(BaseModel):
    genetics: float | None = None
    ppi_proximity: float | None = None
    pathway_enrichment: float | None = None
    safety_off_tissue: float | None = None
    modality_fit: Dict[str, float] | None = None


class EvidenceRefLegacy(BaseModel):
    """Legacy evidence reference model for backward compatibility."""
    label: str = Field(..., description="Display label for the evidence")
    url: str = Field(..., description="URL to external resource")
    type: str = Field(..., description="Type of evidence (literature, database, proprietary)")


class ChannelContribution(BaseModel):
    """Individual channel contribution to final score."""
    channel: str = Field(..., description="Channel name (genetics, ppi, etc.)")
    weight: float = Field(..., ge=0, le=1, description="Weight used for this channel")
    score: Optional[float] = Field(None, ge=0, le=1, description="Raw channel score (None if unavailable)")
    contribution: float = Field(..., ge=0, description="Weighted contribution (weight × score)")
    available: bool = Field(..., description="Whether this channel had valid data")


class Explanation(BaseModel):
    """Comprehensive explanation for target ranking with actionable insights."""
    target: str = Field(..., description="Target gene symbol")
    contributions: List[ChannelContribution] = Field(..., description="Channel contributions sorted by impact")
    evidence_refs: List[EvidenceRefLegacy] = Field(..., description="Clickable evidence references")
    total_weighted_score: float = Field(..., ge=0, le=1, description="Sum of all weighted contributions")
    confidence_level: Optional[str] = Field(None, description="Overall confidence (low/medium/high)")
    key_insights: Optional[List[str]] = Field(None, description="Key biological insights")


class RankImpact(BaseModel):
    """Analysis of ranking changes between weight configurations."""
    target: str = Field(..., description="Target gene symbol")
    rank_baseline: int = Field(..., ge=1, description="Rank with default weights")
    rank_current: int = Field(..., ge=1, description="Rank with current weights")
    delta: int = Field(..., description="Change in rank (positive = moved up)")
    movement: str = Field(..., description="Direction of rank change")
    score_change: Optional[float] = Field(None, description="Change in total score")

    @field_validator('movement')
    @classmethod
    def validate_movement(cls, v):
        """Validate movement direction."""
        if v not in ['up', 'down', 'unchanged']:
            raise ValueError("Movement must be 'up', 'down', or 'unchanged'")
        return v


class CacheMetadata(BaseModel):
    """Cache and performance metadata for API calls."""
    cached: bool = Field(..., description="Whether any responses came from cache")
    fetch_ms: float = Field(..., ge=0, description="Total time spent fetching data")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Proportion of cached responses")
    total_calls: int = Field(..., ge=0, description="Total number of external API calls")
    min_fetch_ms: Optional[float] = Field(None, ge=0, description="Fastest individual call")
    max_fetch_ms: Optional[float] = Field(None, ge=0, description="Slowest individual call")
    avg_fetch_ms: Optional[float] = Field(None, ge=0, description="Average call time")


class TargetScore(BaseModel):
    target: str
    total_score: float
    breakdown: TargetBreakdown
    evidence_refs: List[str] = Field(default_factory=list)
    data_version: str = "Unknown"
    explanation: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    warnings: Optional[List[str]] = None
    channels: Optional[Dict[str, ChannelScore]] = Field(
        default=None,
        description="Per-channel detailed scores, components, evidence and quality"
    )
    model_config = ConfigDict(extra="allow")




class RequestSummary(BaseModel):
    """Summary of the scoring request."""
    disease: str = Field(..., description="Disease identifier used")
    target_count: int = Field(..., ge=1, description="Number of targets scored")
    weights_used: Dict[str, float] = Field(..., description="Final weights applied")
    timestamp: float = Field(..., description="Request timestamp")
    user_id: Optional[str] = Field(None, description="User identifier")


class ScoreResponse(BaseModel):
    """Complete response from target scoring endpoint."""
    targets: List[TargetScore] = Field(..., description="Scored targets with explanations")
    request_summary: RequestSummary = Field(..., description="Summary of request parameters")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time in milliseconds")
    data_version: str = Field(..., description="Consolidated data version string")
    meta: CacheMetadata = Field(..., description="Cache and performance metadata")
    rank_impact: Optional[List[RankImpact]] = Field(None, description="Ranking change analysis")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")
    warnings: Optional[List[str]] = Field(None, description="System-level warnings")


class HealthCheckResponse(BaseModel):
    """Health check endpoint response."""
    status: str = Field(..., description="Health status")
    timestamp: float = Field(...)
    service: str = Field(...)
    version: str = Field(...)
    checks: Optional[Dict[str, Any]] = Field(None, description="Health check results - mixed types allowed")

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate health status."""
        if v not in ['healthy', 'degraded', 'unhealthy']:
            raise ValueError("Status must be 'healthy', 'degraded', or 'unhealthy'")
        return v


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ExportRequest(BaseModel):
    """Request for data export functionality."""
    format: str = Field(..., description="Export format")
    include_explanations: bool = Field(True, description="Include explanation objects")
    include_metadata: bool = Field(True, description="Include metadata")
    filter_targets: Optional[List[str]] = Field(None, description="Export only these targets")

    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        """Validate export format."""
        if v not in ['json', 'csv', 'xlsx']:
            raise ValueError("Format must be 'json', 'csv', or 'xlsx'")
        return v


class SystemSummaryResponse(BaseModel):
    """System capabilities and data source summary."""
    system_info: Dict[str, str] = Field(..., description="Basic system information")
    scoring_channels: Dict[str, str] = Field(..., description="Available scoring channels")
    default_weights: Dict[str, float] = Field(..., description="Default channel weights")
    features: List[str] = Field(..., description="System capabilities")
    data_sources: List[str] = Field(..., description="Data source descriptions")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Usage statistics")


# ========================
# Internal Scoring Models
# ========================

class ChannelResult(BaseModel):
    """Internal model for channel scoring results."""
    score: Optional[float] = Field(None, ge=0, le=1)
    evidence_refs: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)


class ScoringContext(BaseModel):
    """Internal context for scoring operations."""
    disease: str
    targets: List[str]
    weights: Dict[str, float]
    request_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None


# ========================
# Utility Functions
# ========================

def serialize_explanation(explanation: Union[Explanation, Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize explanation object for JSON response."""
    if isinstance(explanation, Explanation):
        return explanation.model_dump()
    return explanation
# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Pydantic schemas for VantAI Target Scoreboard API - Phase 1.
Kanonik modeller: gerçek veri entegrasyonu için production-ready şemalar.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any, Union, Literal
from datetime import datetime
import re


def schema_version() -> str:
    """Return repo-scoped schema version."""
    return "v1.0.0-phase1"


def get_utc_now() -> datetime:
    """Consistent UTC timestamp helper."""
    return datetime.utcnow()


# ========================
# Core Data Quality Models
# ========================

class DataQualityFlags(BaseModel):
    """Data quality indicators for evidence tracking."""
    stale: bool = Field(False, description="Data older than acceptable threshold")
    partial: bool = Field(False, description="Incomplete data returned")
    schema_version: str = Field(default_factory=schema_version, description="Schema version used")
    notes: Optional[str] = Field(None, description="Additional quality notes")


class ValidationResult(BaseModel):
    """Result of data validation process."""
    ok: bool = Field(..., description="Whether validation passed")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    quality: DataQualityFlags = Field(default_factory=DataQualityFlags, description="Quality flags")


# ========================
# Evidence and Reference Models
# ========================

class EvidenceRef(BaseModel):
    """Evidence reference with external links and quality metadata."""
    source: str = Field(..., description="Data source identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID if literature evidence")
    title: Optional[str] = Field(None, description="Article or evidence title")
    journal: Optional[str] = Field(None, description="Journal name for literature")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Publication year")
    url: Optional[str] = Field(None, description="Direct URL to evidence")
    source_quality: Optional[str] = Field(None, description="Source quality assessment")
    timestamp: datetime = Field(default_factory=get_utc_now, description="When evidence was retrieved")


# ========================
# Biological Data Models
# ========================

class AssociationRecord(BaseModel):
    """Gene-disease association from external databases."""
    gene: str = Field(..., description="Gene symbol")
    disease: str = Field(..., description="Disease identifier")
    score: float = Field(..., ge=0, le=1, description="Association strength score")
    pval: Optional[float] = Field(None, ge=0, le=1, description="Statistical p-value if available")
    source: str = Field(..., description="Source database")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Retrieval timestamp")
    evidence: List[EvidenceRef] = Field(default_factory=list, description="Supporting evidence")


class PPIEdge(BaseModel):
    """Protein-protein interaction edge."""
    source_gene: str = Field(..., description="Source gene symbol")
    partner: str = Field(..., description="Interaction partner gene symbol")
    confidence: float = Field(..., ge=0, le=1, description="Interaction confidence score")
    evidence: List[EvidenceRef] = Field(default_factory=list, description="Supporting evidence")


class PPINetwork(BaseModel):
    """Complete protein-protein interaction network for a gene."""
    gene: str = Field(..., description="Central gene symbol")
    edges: List[PPIEdge] = Field(default_factory=list, description="Network edges")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Network retrieval time")
    source: str = Field(..., description="Network data source")


class ExpressionRecord(BaseModel):
    """Gene expression measurement."""
    gene: str = Field(..., description="Gene symbol")
    tissue: str = Field(..., description="Tissue type")
    value: float = Field(..., description="Expression value")
    unit: Optional[str] = Field(None, description="Expression unit (TPM, FPKM, etc.)")
    quantile: Optional[float] = Field(None, ge=0, le=1, description="Expression quantile in tissue")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Measurement timestamp")
    source: str = Field(..., description="Expression data source")


class StructureConfidence(BaseModel):
    """Protein structure confidence scores."""
    gene: str = Field(..., description="Gene symbol")
    plddt_mean: Optional[float] = Field(None, ge=0, le=100, description="Mean pLDDT confidence")
    pae_mean: Optional[float] = Field(None, ge=0, description="Mean PAE (predicted aligned error)")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Structure data timestamp")
    source: str = Field(..., description="Structure database source")


# ========================
# Channel Scoring Models
# ========================

class ChannelScore(BaseModel):
    """Individual channel contribution to target score."""
    name: str = Field(..., description="Channel identifier")
    score: Optional[float] = Field(None, ge=0, le=1, description="Channel score (None if data missing)")
    status: Literal["ok", "data_missing", "error"] = Field(..., description="Channel status")
    components: Dict[str, Any] = Field(default_factory=dict, description="Sub-component scores - mixed types")
    evidence: List[EvidenceRef] = Field(default_factory=list, description="Supporting evidence")
    quality: DataQualityFlags = Field(default_factory=DataQualityFlags, description="Data quality flags")

class TargetScoreBundle(BaseModel):
    """Complete scoring bundle for a target-disease pair."""
    gene: str = Field(..., description="Target gene symbol")
    disease: Optional[str] = Field(None, description="Disease identifier")
    channels: Dict[str, ChannelScore] = Field(default_factory=dict, description="Channel scores")
    combined_score: Optional[float] = Field(None, ge=0, le=1, description="Weighted combined score")
    lineage: Dict[str, Any] = Field(default_factory=dict, description="Data provenance tracking")
    timestamp: datetime = Field(default_factory=get_utc_now, description="Scoring timestamp")


# ========================
# API Request/Response Models (keeping existing for compatibility)
# ========================

class ScoreRequest(BaseModel):
    """Request model for target scoring endpoint."""
    disease: str = Field(..., description="Disease identifier (EFO, MONDO, etc.)")
    targets: List[str] = Field(..., min_length=1, max_length=50, description="List of target gene symbols")
    weights: Dict[str, float] = Field(
        default={
            "genetics": 0.35,
            "ppi": 0.25,
            "pathway": 0.20,
            "safety": 0.10,
            "modality_fit": 0.10
        },
        description="Channel weights for scoring (must sum to ~1.0)"
    )

    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Validate that weights are reasonable."""
        if not isinstance(v, dict):
            raise ValueError("Weights must be a dictionary")

        # Check individual weight bounds
        for channel, weight in v.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for {channel} must be between 0 and 1")

        # Check total weight sum
        total_weight = sum(v.values())
        if not 0.8 <= total_weight <= 1.2:
            raise ValueError(f"Total weight sum {total_weight:.2f} should be approximately 1.0")

        return v

    @field_validator('targets')
    @classmethod
    def validate_targets(cls, v):
        """Validate target gene symbols."""
        if not v:
            raise ValueError("At least one target must be provided")

        # Clean and validate gene symbols
        cleaned_targets = []
        for target in v:
            if not isinstance(target, str):
                raise ValueError("All targets must be strings")

            cleaned = target.strip().upper()
            if len(cleaned) < 2:
                raise ValueError(f"Target '{target}' is too short")

            if not cleaned.replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"Target '{target}' contains invalid characters")

            cleaned_targets.append(cleaned)

        return cleaned_targets


class ModalityFitScores(BaseModel):
    """Detailed modality fit subscores."""
    e3_coexpr: float = Field(..., ge=0, le=1, description="E3 ligase co-expression score")
    ternary_proxy: float = Field(..., ge=0, le=1, description="Ternary complex formation proxy")
    ppi_hotspot: float = Field(..., ge=0, le=1, description="PPI hotspot druggability score")
    overall_druggability: float = Field(..., ge=0, le=1, description="Combined modality fitness")
    protac_degrader: float = Field(..., ge=0, le=1, description="PROTAC/degrader suitability")
    small_molecule: float = Field(..., ge=0, le=1, description="Small molecule druggability")
    molecular_glue: Optional[float] = Field(None, ge=0, le=1, description="Molecular glue potential")
    adc_score: Optional[float] = Field(None, ge=0, le=1, description="ADC targeting potential")


class TargetBreakdown(BaseModel):
    genetics: float | None = None
    ppi_proximity: float | None = None
    pathway_enrichment: float | None = None
    safety_off_tissue: float | None = None
    modality_fit: Dict[str, float] | None = None


class EvidenceRefLegacy(BaseModel):
    """Legacy evidence reference model for backward compatibility."""
    label: str = Field(..., description="Display label for the evidence")
    url: str = Field(..., description="URL to external resource")
    type: str = Field(..., description="Type of evidence (literature, database, proprietary)")


class ChannelContribution(BaseModel):
    """Individual channel contribution to final score."""
    channel: str = Field(..., description="Channel name (genetics, ppi, etc.)")
    weight: float = Field(..., ge=0, le=1, description="Weight used for this channel")
    score: Optional[float] = Field(None, ge=0, le=1, description="Raw channel score (None if unavailable)")
    contribution: float = Field(..., ge=0, description="Weighted contribution (weight × score)")
    available: bool = Field(..., description="Whether this channel had valid data")


class Explanation(BaseModel):
    """Comprehensive explanation for target ranking with actionable insights."""
    target: str = Field(..., description="Target gene symbol")
    contributions: List[ChannelContribution] = Field(..., description="Channel contributions sorted by impact")
    evidence_refs: List[EvidenceRefLegacy] = Field(..., description="Clickable evidence references")
    total_weighted_score: float = Field(..., ge=0, le=1, description="Sum of all weighted contributions")
    confidence_level: Optional[str] = Field(None, description="Overall confidence (low/medium/high)")
    key_insights: Optional[List[str]] = Field(None, description="Key biological insights")


class RankImpact(BaseModel):
    """Analysis of ranking changes between weight configurations."""
    target: str = Field(..., description="Target gene symbol")
    rank_baseline: int = Field(..., ge=1, description="Rank with default weights")
    rank_current: int = Field(..., ge=1, description="Rank with current weights")
    delta: int = Field(..., description="Change in rank (positive = moved up)")
    movement: str = Field(..., description="Direction of rank change")
    score_change: Optional[float] = Field(None, description="Change in total score")

    @field_validator('movement')
    @classmethod
    def validate_movement(cls, v):
        """Validate movement direction."""
        if v not in ['up', 'down', 'unchanged']:
            raise ValueError("Movement must be 'up', 'down', or 'unchanged'")
        return v


class CacheMetadata(BaseModel):
    """Cache and performance metadata for API calls."""
    cached: bool = Field(..., description="Whether any responses came from cache")
    fetch_ms: float = Field(..., ge=0, description="Total time spent fetching data")
    cache_hit_rate: float = Field(..., ge=0, le=1, description="Proportion of cached responses")
    total_calls: int = Field(..., ge=0, description="Total number of external API calls")
    min_fetch_ms: Optional[float] = Field(None, ge=0, description="Fastest individual call")
    max_fetch_ms: Optional[float] = Field(None, ge=0, description="Slowest individual call")
    avg_fetch_ms: Optional[float] = Field(None, ge=0, description="Average call time")


class TargetScore(BaseModel):
    target: str
    total_score: float
    breakdown: TargetBreakdown
    evidence_refs: List[str] = Field(default_factory=list)
    data_version: str = "Unknown"
    explanation: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    warnings: Optional[List[str]] = None

    class Config:
        extra = "allow"


class RequestSummary(BaseModel):
    """Summary of the scoring request."""
    disease: str = Field(..., description="Disease identifier used")
    target_count: int = Field(..., ge=1, description="Number of targets scored")
    weights_used: Dict[str, float] = Field(..., description="Final weights applied")
    timestamp: float = Field(..., description="Request timestamp")
    user_id: Optional[str] = Field(None, description="User identifier")


class ScoreResponse(BaseModel):
    """Complete response from target scoring endpoint."""
    targets: List[TargetScore] = Field(..., description="Scored targets with explanations")
    request_summary: RequestSummary = Field(..., description="Summary of request parameters")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time in milliseconds")
    data_version: str = Field(..., description="Consolidated data version string")
    meta: CacheMetadata = Field(..., description="Cache and performance metadata")
    rank_impact: Optional[List[RankImpact]] = Field(None, description="Ranking change analysis")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")
    warnings: Optional[List[str]] = Field(None, description="System-level warnings")


class HealthCheckResponse(BaseModel):
    """Health check endpoint response."""
    status: str = Field(..., description="Health status")
    timestamp: float = Field(...)
    service: str = Field(...)
    version: str = Field(...)
    checks: Optional[Dict[str, Any]] = Field(None, description="Health check results - mixed types allowed")
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        """Validate health status."""
        if v not in ['healthy', 'degraded', 'unhealthy']:
            raise ValueError("Status must be 'healthy', 'degraded', or 'unhealthy'")
        return v


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ExportRequest(BaseModel):
    """Request for data export functionality."""
    format: str = Field(..., description="Export format")
    include_explanations: bool = Field(True, description="Include explanation objects")
    include_metadata: bool = Field(True, description="Include metadata")
    filter_targets: Optional[List[str]] = Field(None, description="Export only these targets")

    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        """Validate export format."""
        if v not in ['json', 'csv', 'xlsx']:
            raise ValueError("Format must be 'json', 'csv', or 'xlsx'")
        return v


class SystemSummaryResponse(BaseModel):
    """System capabilities and data source summary."""
    system_info: Dict[str, str] = Field(..., description="Basic system information")
    scoring_channels: Dict[str, str] = Field(..., description="Available scoring channels")
    default_weights: Dict[str, float] = Field(..., description="Default channel weights")
    features: List[str] = Field(..., description="System capabilities")
    data_sources: List[str] = Field(..., description="Data source descriptions")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Usage statistics")


# ========================
# Internal Scoring Models
# ========================

class ChannelResult(BaseModel):
    """Internal model for channel scoring results."""
    score: Optional[float] = Field(None, ge=0, le=1)
    evidence_refs: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)


class ScoringContext(BaseModel):
    """Internal context for scoring operations."""
    disease: str
    targets: List[str]
    weights: Dict[str, float]
    request_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None


# ========================
# Utility Functions
# ========================

def serialize_explanation(explanation: Union[Explanation, Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize explanation object for JSON response."""
    if isinstance(explanation, Explanation):
        return explanation.model_dump()
    return explanation


def validate_gene_symbol(symbol: str) -> str:
    """Validate and normalize gene symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Gene symbol must be a non-empty string")

    cleaned = symbol.strip().upper()
    if len(cleaned) < 2:
        raise ValueError("Gene symbol too short")

    if not cleaned.replace('_', '').replace('-', '').isalnum():
        raise ValueError("Gene symbol contains invalid characters")

    return cleaned

def validate_gene_symbol(symbol: str) -> str:
    """Validate and normalize gene symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Gene symbol must be a non-empty string")

    cleaned = symbol.strip().upper()
    if len(cleaned) < 2:
        raise ValueError("Gene symbol too short")

    if not cleaned.replace('_', '').replace('-', '').isalnum():
        raise ValueError("Gene symbol contains invalid characters")

    return cleaned