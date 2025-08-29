# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

"""
Data quality validation for VantAI Target Scoreboard.
Validates all incoming biological data for completeness, freshness, and format correctness.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .schemas import (
    ValidationResult,
    DataQualityFlags,
    AssociationRecord,
    PPINetwork,
    PPIEdge,
    ExpressionRecord,
    StructureConfidence
)

# Configure structured logger
logger = logging.getLogger(__name__)

# Configuration from environment
STALENESS_THRESHOLDS = {
    "opentargets": int(os.getenv("OT_STALENESS_HOURS", "48")),      # 48h default
    "stringdb": int(os.getenv("STRING_STALENESS_HOURS", "168")),    # 7 days default
    "expression_atlas": int(os.getenv("ATLAS_STALENESS_HOURS", "168")),  # 7 days
    "alphafold": int(os.getenv("ALPHAFOLD_STALENESS_HOURS", "168")),     # 7 days  
    "pubmed": int(os.getenv("PUBMED_STALENESS_HOURS", "168")),           # 7 days
    "default": int(os.getenv("DEFAULT_STALENESS_HOURS", "72"))           # 3 days fallback
}

MAX_LIST_LENGTH = int(os.getenv("MAX_VALIDATION_LIST_LENGTH", "1000"))
MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE_SCORE", "0.0"))
MAX_CONFIDENCE_SCORE = float(os.getenv("MAX_CONFIDENCE_SCORE", "1.0"))


class DataQualityValidator:
    """
    Validates biological data for quality, completeness, and freshness.
    Each validation returns a ValidationResult with specific issues identified.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.logger = logger

    def validate(self, source: str, payload: Any) -> ValidationResult:
        """
        Main validation entry point. Routes to appropriate validator based on source.
        
        Args:
            source: Data source identifier (opentargets, stringdb, etc.)
            payload: Data to validate
            
        Returns:
            ValidationResult with ok/issues/quality flags
        """
        issues = []
        quality = DataQualityFlags()

        # Basic payload checks
        if payload is None:
            issues.append("Payload is None")
            return ValidationResult(ok=False, issues=issues, quality=quality)

        if not isinstance(payload, (dict, list)) and not hasattr(payload, '__dict__'):
            issues.append(f"Payload type {type(payload)} not supported for validation")
            return ValidationResult(ok=False, issues=issues, quality=quality)

        # Source-specific validation
        try:
            if source.lower() == "opentargets":
                result = self.validate_associations(payload)
            elif source.lower() == "stringdb":
                result = self.validate_ppi(payload)
            elif source.lower() in ("expression_atlas", "atlas"):
                result = self.validate_expression(payload)
            elif source.lower() == "alphafold":
                result = self.validate_structure(payload)
            else:
                # Generic validation for unknown sources
                result = self._validate_generic(payload, source)

            # Check staleness
            staleness_result = self._check_staleness(payload, source)
            if staleness_result.issues:
                result.issues.extend(staleness_result.issues)
                result.quality.stale = staleness_result.quality.stale

            # Log validation results
            if result.issues:
                self.logger.warning(
                    f"Validation issues for {source}",
                    extra={
                        "source": source,
                        "issues": result.issues,
                        "quality_flags": result.quality.model_dump()
                    }
                )

            return result

        except Exception as e:
            self.logger.error(f"Validation error for {source}: {e}", extra={"source": source})
            issues.append(f"Validation exception: {str(e)}")
            return ValidationResult(ok=False, issues=issues, quality=quality)

    def validate_associations(self, data: Union[Dict, List, AssociationRecord]) -> ValidationResult:
        """Validate OpenTargets association data."""
        issues = []
        quality = DataQualityFlags()

        if isinstance(data, AssociationRecord):
            # Pydantic model validation
            return self._validate_association_record(data)

        if isinstance(data, dict):
            # Raw dict from API
            issues.extend(self._check_required_fields(data, ["gene", "disease", "score"]))
            
            # Score validation
            score = data.get("score")
            if score is not None:
                if not isinstance(score, (int, float)):
                    issues.append("Score must be numeric")
                elif not (0.0 <= score <= 1.0):
                    issues.append(f"Score {score} outside valid range [0,1]")

            # Evidence count check
            evidence_count = data.get("evidence_count", 0)
            if evidence_count == 0:
                quality.partial = True
                issues.append("No evidence count provided")

        elif isinstance(data, list):
            # List of associations
            if len(data) == 0:
                quality.partial = True
                issues.append("Empty association list")
            elif len(data) > MAX_LIST_LENGTH:
                issues.append(f"Association list too long: {len(data)} > {MAX_LIST_LENGTH}")
            else:
                # Validate first few items
                for i, item in enumerate(data[:5]):
                    item_result = self.validate_associations(item)
                    if item_result.issues:
                        issues.extend([f"Item {i}: {issue}" for issue in item_result.issues])

        return ValidationResult(
            ok=len(issues) == 0 or (not self.strict_mode and not any("must be" in issue for issue in issues)),
            issues=issues,
            quality=quality
        )

    def validate_ppi(self, data: Union[Dict, PPINetwork]) -> ValidationResult:
        """Validate STRING-DB protein-protein interaction data."""
        issues = []
        quality = DataQualityFlags()

        if isinstance(data, PPINetwork):
            # Validate network structure
            if len(data.edges) == 0:
                quality.partial = True
                issues.append("Empty PPI network")
            
            # Check edge quality
            low_confidence_count = sum(1 for edge in data.edges if edge.confidence < 0.4)
            if low_confidence_count > len(data.edges) * 0.8:
                quality.partial = True
                issues.append(f"Most edges have low confidence: {low_confidence_count}/{len(data.edges)}")

        elif isinstance(data, dict):
            issues.extend(self._check_required_fields(data, ["gene", "edges"]))
            
            edges = data.get("edges", [])
            if not isinstance(edges, list):
                issues.append("Edges must be a list")
            elif len(edges) == 0:
                quality.partial = True
                issues.append("No PPI edges found")
            else:
                # Validate edge structure
                for i, edge in enumerate(edges[:10]):  # Check first 10 edges
                    if not isinstance(edge, dict):
                        issues.append(f"Edge {i}: must be dict")
                        continue
                    
                    edge_issues = self._check_required_fields(edge, ["partner", "confidence"])
                    if edge_issues:
                        issues.extend([f"Edge {i}: {issue}" for issue in edge_issues])
                    
                    conf = edge.get("confidence")
                    if conf is not None and not (0.0 <= conf <= 1.0):
                        issues.append(f"Edge {i}: confidence {conf} outside [0,1]")

        return ValidationResult(
            ok=len(issues) == 0 or not self.strict_mode,
            issues=issues,
            quality=quality
        )

    def validate_expression(self, data: Union[Dict, List, ExpressionRecord]) -> ValidationResult:
        """Validate Expression Atlas data."""
        issues = []
        quality = DataQualityFlags()

        if isinstance(data, ExpressionRecord):
            # Validate expression value
            if data.value < 0:
                issues.append(f"Negative expression value: {data.value}")
            
            return ValidationResult(ok=len(issues) == 0, issues=issues, quality=quality)

        elif isinstance(data, dict):
            issues.extend(self._check_required_fields(data, ["gene", "tissue", "value"]))
            
            value = data.get("value")
            if value is not None:
                if not isinstance(value, (int, float)):
                    issues.append("Expression value must be numeric")
                elif value < 0:
                    issues.append(f"Negative expression value: {value}")

        elif isinstance(data, list):
            if len(data) == 0:
                quality.partial = True
                issues.append("No expression data found")

        return ValidationResult(
            ok=len(issues) == 0 or not self.strict_mode,
            issues=issues,
            quality=quality
        )

    def validate_structure(self, data: Union[Dict, StructureConfidence]) -> ValidationResult:
        """Validate AlphaFold structure confidence data."""
        issues = []
        quality = DataQualityFlags()

        if isinstance(data, StructureConfidence):
            # Validate confidence scores
            if data.plddt_mean is not None and not (0 <= data.plddt_mean <= 100):
                issues.append(f"pLDDT mean {data.plddt_mean} outside [0,100]")
            
            if data.pae_mean is not None and data.pae_mean < 0:
                issues.append(f"PAE mean {data.pae_mean} cannot be negative")

        elif isinstance(data, dict):
            issues.extend(self._check_required_fields(data, ["gene"]))
            
            plddt = data.get("plddt_mean")
            if plddt is not None:
                if not isinstance(plddt, (int, float)):
                    issues.append("pLDDT must be numeric")
                elif not (0 <= plddt <= 100):
                    issues.append(f"pLDDT {plddt} outside valid range [0,100]")

            pae = data.get("pae_mean")
            if pae is not None:
                if not isinstance(pae, (int, float)):
                    issues.append("PAE must be numeric")
                elif pae < 0:
                    issues.append(f"PAE {pae} cannot be negative")

            # Check if any structure data present
            if plddt is None and pae is None:
                quality.partial = True
                issues.append("No structure confidence data available")

        return ValidationResult(
            ok=len(issues) == 0 or not self.strict_mode,
            issues=issues,
            quality=quality
        )

    def _validate_association_record(self, record: AssociationRecord) -> ValidationResult:
        """Validate individual association record."""
        issues = []
        quality = DataQualityFlags()

        # Check score bounds
        if not (0.0 <= record.score <= 1.0):
            issues.append(f"Association score {record.score} outside [0,1]")

        # Check p-value if present
        if record.pval is not None and not (0.0 <= record.pval <= 1.0):
            issues.append(f"P-value {record.pval} outside [0,1]")

        # Check evidence
        if len(record.evidence) == 0:
            quality.partial = True
            issues.append("No evidence references provided")

        return ValidationResult(
            ok=len(issues) == 0,
            issues=issues,
            quality=quality
        )

    def _validate_generic(self, payload: Any, source: str) -> ValidationResult:
        """Generic validation for unknown data sources."""
        issues = []
        quality = DataQualityFlags()

        # Basic structure checks
        if isinstance(payload, dict):
            if len(payload) == 0:
                quality.partial = True
                issues.append("Empty data dictionary")
            
            # Check for timestamp
            if "timestamp" not in payload:
                quality.notes = f"No timestamp in {source} data"

        elif isinstance(payload, list):
            if len(payload) == 0:
                quality.partial = True
                issues.append("Empty data list")
            elif len(payload) > MAX_LIST_LENGTH:
                issues.append(f"Data list too long: {len(payload)} > {MAX_LIST_LENGTH}")

        return ValidationResult(
            ok=len(issues) == 0,
            issues=issues,
            quality=quality
        )

    def _check_staleness(self, payload: Any, source: str) -> ValidationResult:
        """Check if data is stale based on timestamp."""
        issues = []
        quality = DataQualityFlags()

        # Get staleness threshold for source
        threshold_hours = STALENESS_THRESHOLDS.get(source.lower(), STALENESS_THRESHOLDS["default"])
        threshold = timedelta(hours=threshold_hours)

        # Extract timestamp from payload
        timestamp = None
        
        if hasattr(payload, 'timestamp'):
            timestamp = payload.timestamp
        elif isinstance(payload, dict):
            ts_value = payload.get("timestamp")
            if ts_value:
                if isinstance(ts_value, datetime):
                    timestamp = ts_value
                elif isinstance(ts_value, str):
                    try:
                        timestamp = datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                    except ValueError:
                        issues.append(f"Invalid timestamp format: {ts_value}")

        if timestamp is None:
            quality.notes = f"No timestamp found for staleness check"
        else:
            age = datetime.utcnow() - timestamp.replace(tzinfo=None)
            if age > threshold:
                quality.stale = True
                issues.append(f"Data is stale: {age.days}d {age.seconds//3600}h old (threshold: {threshold_hours}h)")

        return ValidationResult(
            ok=len(issues) == 0,
            issues=issues,
            quality=quality
        )

    def _check_required_fields(self, data: Dict, required_fields: List[str]) -> List[str]:
        """Check if required fields are present and non-empty."""
        issues = []
        
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
            elif data[field] is None:
                issues.append(f"Required field {field} is None")
            elif isinstance(data[field], str) and not data[field].strip():
                issues.append(f"Required field {field} is empty string")

        return issues

    def validate_batch(self, source: str, payload_list: List[Any]) -> Dict[str, ValidationResult]:
        """
        Validate a batch of payloads from the same source.
        
        Returns:
            Dict mapping payload index to ValidationResult
        """
        results = {}
        
        for i, payload in enumerate(payload_list):
            try:
                results[str(i)] = self.validate(source, payload)
            except Exception as e:
                self.logger.error(f"Batch validation error for {source}[{i}]: {e}")
                results[str(i)] = ValidationResult(
                    ok=False,
                    issues=[f"Validation exception: {str(e)}"],
                    quality=DataQualityFlags()
                )

        return results

    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Create summary statistics from validation results.
        
        Args:
            results: Dict of validation results
            
        Returns:
            Summary with pass/fail counts and common issues
        """
        total = len(results)
        passed = sum(1 for r in results.values() if r.ok)
        failed = total - passed

        # Collect all issues
        all_issues = []
        stale_count = 0
        partial_count = 0

        for result in results.values():
            all_issues.extend(result.issues)
            if result.quality.stale:
                stale_count += 1
            if result.quality.partial:
                partial_count += 1

        # Count issue types
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        return {
            "total_validated": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "stale_data_count": stale_count,
            "partial_data_count": partial_count,
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "total_issues": len(all_issues)
        }


# ========================
# Global validator instance
# ========================

_validator: Optional[DataQualityValidator] = None


def get_validator(strict_mode: bool = False) -> DataQualityValidator:
    """Get global validator instance."""
    global _validator
    if _validator is None or _validator.strict_mode != strict_mode:
        _validator = DataQualityValidator(strict_mode=strict_mode)
    return _validator


def validate_data(source: str, payload: Any, strict: bool = False) -> ValidationResult:
    """Convenience function for quick validation."""
    validator = get_validator(strict_mode=strict)
    return validator.validate(source, payload)