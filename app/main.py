# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.


from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import time
import logging
import json
import csv
import io
from datetime import datetime
from typing import List
from collections import defaultdict

from .schemas import ScoreRequest, ScoreResponse, TargetScore
from .scoring import score_targets, validate_score_request, target_scorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("VantAI Target Scoreboard API starting up...")
    logger.info("Loading PPI network and pathway data...")

    # Warm up the scoring system (optional)
    try:
        # Test scoring with a dummy request to warm up caches
        dummy_request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR"],
            weights={"genetics": 0.5, "ppi": 0.5}
        )
        target_scores, metadata = await score_targets(dummy_request)
        logger.info("Scoring system warmed up successfully")
        logger.info(f"Warmup metadata: {metadata}")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

    yield

    # Shutdown
    logger.info("VantAI Target Scoreboard API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="VantAI Target Scoreboard",
    description="Modality-aware target scoring system built on Open Targets data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "VantAI Target Scoreboard",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "VantAI Target Scoreboard API",
        "version": "1.0.0",
        "description": "Modality-aware target scoring system",
        "endpoints": {
            "health": "/healthz",
            "score": "/score",
            "export": "/export/{format}",
            "docs": "/docs"
        },
        "example_usage": {
            "curl": "curl -X POST http://localhost:8000/score -H 'Content-Type: application/json' -d @examples/sample_payload.json"
        }
    }


@app.post("/score", response_model=ScoreResponse)
async def score_targets_endpoint(request: ScoreRequest):
    """
    Score targets for a given disease using multi-modal approach.

    This endpoint combines:
    - Open Targets genetics associations
    - PPI network proximity with RWR
    - Pathway enrichment
    - VantAI proprietary modality fit scoring
    - Safety considerations

    Returns detailed breakdown with evidence references, data versioning,
    actionable explanations, and ranking impact analysis.
    """
    start_time = time.time()

    # Validate request
    is_valid, error_msg = validate_score_request(request)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        logger.info(f"Scoring request: disease={request.disease}, targets={request.targets}")

        # Score targets with metadata and explanations
        target_scores, scoring_metadata = await score_targets(request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build enhanced response
        response = ScoreResponse(
            targets=target_scores,
            request_summary={
                "disease": request.disease,
                "target_count": len(request.targets),
                "weights_used": request.weights,
                "timestamp": time.time()
            },
            processing_time_ms=processing_time_ms,
            data_version=scoring_metadata.get("data_version", "Unknown"),
            meta=scoring_metadata.get("meta", {"cached": False, "fetch_ms": 0.0}),
            rank_impact=scoring_metadata.get("rank_impact", [])
        )

        # Enhanced logging with explanation and ranking info
        cache_meta = scoring_metadata.get("meta", {})
        cache_hit_rate = cache_meta.get("cache_hit_rate", 0.0)
        total_fetch_ms = cache_meta.get("fetch_ms", 0.0)
        rank_changes = len(
            [r for r in response.rank_impact if r.movement != "unchanged"]) if response.rank_impact else 0

        logger.info(
            f"Scoring completed in {processing_time_ms:.1f}ms for {len(target_scores)} targets. "
            f"Cache hit rate: {cache_hit_rate:.2%}, Total fetch time: {total_fetch_ms:.1f}ms, "
            f"Ranking changes: {rank_changes}/{len(target_scores)}, "
            f"Data version: {response.data_version}"
        )

        return response

    except Exception as e:
        logger.error(f"Error processing score request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during scoring: {str(e)}"
        )


@app.get("/targets/{target}/explanation")
async def get_target_explanation(target: str, disease: str = "EFO_0000305",
                                 weights: str = None):
    """
    Get detailed explanation for a single target's ranking.

    Args:
        target: Target gene symbol
        disease: Disease EFO ID
        weights: JSON string of weights (optional)

    Returns:
        Detailed explanation with contributions and evidence
    """
    try:
        import json

        # Parse weights if provided
        if weights:
            try:
                weight_dict = json.loads(weights)
            except json.JSONDecodeError:
                weight_dict = {}
        else:
            weight_dict = {}

        # Create single-target request
        request = ScoreRequest(
            disease=disease,
            targets=[target],
            weights=weight_dict
        )

        # Score the target
        target_scores, metadata = await score_targets(request)

        if not target_scores:
            raise HTTPException(status_code=404, detail=f"Target {target} not found or could not be scored")

        target_score = target_scores[0]
        explanation = target_score.explanation

        return {
            "target": target,
            "explanation": explanation,
            "metadata": metadata,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting explanation for {target}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation: {str(e)}"
        )


@app.post("/export/{format}")
async def export_results(format: str, request: ScoreRequest):
    """
    Export scoring results in JSON or CSV format with explanations.
    """
    if format not in ['json', 'csv']:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

    try:
        # Get scoring results with explanations and ranking
        target_scores, scoring_metadata = await score_targets(request)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disease_short = request.disease.replace("EFO_", "").replace("_", "")

        if format == 'json':
            # JSON export - full data with explanations
            export_data = {
                "metadata": {
                    "export_timestamp": timestamp,
                    "disease": request.disease,
                    "targets_count": len(request.targets),
                    "weights_used": request.weights,
                    "vantai_version": "1.0.0",
                    "data_version": scoring_metadata.get("data_version", "Unknown"),
                    "cache_metadata": scoring_metadata.get("meta", {}),
                    "rank_impact": scoring_metadata.get("rank_impact", [])
                },
                "results": [target.dict() for target in target_scores]
            }

            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            buffer = io.BytesIO(json_str.encode('utf-8'))

            filename = f"vantai_targets_{disease_short}_{timestamp}.json"
            media_type = "application/json"

        else:  # CSV export with ranking info
            buffer = io.StringIO()
            writer = csv.writer(buffer)

            # Header with metadata
            data_version = scoring_metadata.get("data_version", "Unknown")
            writer.writerow([f"# VantAI Target Scoreboard Export - {data_version}"])
            writer.writerow([f"# Export timestamp: {timestamp}"])
            writer.writerow([f"# Disease: {request.disease}"])

            # Enhanced header with ranking info
            writer.writerow([
                'Target', 'Total_Score', 'Current_Rank', 'Baseline_Rank', 'Rank_Delta',
                'Genetics_Score', 'PPI_Proximity', 'Pathway_Enrichment', 'Safety_Score',
                'Modality_Overall', 'Modality_PROTAC', 'Modality_SmallMol',
                'Modality_E3', 'Modality_Ternary', 'Modality_Hotspot',
                'Data_Version', 'Evidence_Count'
            ])

            # Create ranking lookup
            rank_lookup = {}
            if scoring_metadata.get("rank_impact"):
                rank_lookup = {item["target"]: item for item in scoring_metadata["rank_impact"]}

            # Data rows with ranking information
            sorted_targets = sorted(target_scores, key=lambda x: x.total_score, reverse=True)
            for i, ts in enumerate(sorted_targets, 1):
                mf = ts.breakdown.modality_fit or {}
                rank_info = rank_lookup.get(ts.target, {})

                writer.writerow([
                    ts.target,
                    f"{ts.total_score:.4f}",
                    i,  # Current rank
                    rank_info.get("rank_baseline", i),
                    rank_info.get("delta", 0),
                    f"{(ts.breakdown.genetics or 0):.4f}",
                    f"{(ts.breakdown.ppi_proximity or 0):.4f}",
                    f"{(ts.breakdown.pathway_enrichment or 0):.4f}",
                    f"{(ts.breakdown.safety_off_tissue or 0):.4f}",
                    f"{mf.get('overall_druggability', 0):.4f}",
                    f"{mf.get('protac_degrader', 0):.4f}",
                    f"{mf.get('small_molecule', 0):.4f}",
                    f"{mf.get('e3_coexpr', 0):.4f}",
                    f"{mf.get('ternary_proxy', 0):.4f}",
                    f"{mf.get('ppi_hotspot', 0):.4f}",
                    ts.data_version,
                    len(ts.evidence_refs)
                ])

            buffer.seek(0)
            filename = f"vantai_targets_{disease_short}_{timestamp}.csv"
            media_type = "text/csv"

            # Convert to BytesIO
            csv_content = buffer.getvalue()
            buffer = io.BytesIO(csv_content.encode('utf-8'))

        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/export/template/{format}")
async def get_export_template(format: str):
    """
    Get template/sample export file to show format.

    Args:
        format: 'json' or 'csv'
    """
    if format not in ['json', 'csv']:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

    if format == 'json':
        template = {
            "metadata": {
                "export_timestamp": "20250101_120000",
                "disease": "EFO_0000305",
                "targets_count": 2,
                "weights_used": {"genetics": 0.35, "ppi": 0.25},
                "vantai_version": "1.0.0",
                "data_version": "OT-2025.06 | STRING-v12.0 | Reactome-2024",
                "cache_metadata": {
                    "cached": True,
                    "fetch_ms": 45.2,
                    "cache_hit_rate": 0.75,
                    "total_calls": 4
                }
            },
            "results": [
                {
                    "target": "EGFR",
                    "total_score": 0.845,
                    "breakdown": {
                        "genetics": 0.923,
                        "ppi_proximity": 0.756,
                        "pathway_enrichment": 0.689,
                        "safety_off_tissue": 0.234,
                        "modality_fit": {
                            "overall_druggability": 0.812,
                            "protac_degrader": 0.678,
                            "small_molecule": 0.891
                        }
                    },
                    "evidence_refs": ["OpenTargets:2025.06", "PMID:12345678", "STRING:v12.0"],
                    "data_version": "OT-2025.06 | STRING-v12.0 | Reactome-2024"
                }
            ]
        }
        return template

    else:  # CSV template
        template_csv = """# VantAI Target Scoreboard Export - OT-2025.06 | STRING-v12.0 | Reactome-2024
        # Export timestamp: 20250101_120000
        # Disease: EFO_0000305
        Target,Total_Score,Current_Rank,Baseline_Rank,Rank_Delta,Genetics_Score,PPI_Proximity,Pathway_Enrichment,Safety_Score,Modality_Overall,Modality_PROTAC,Modality_SmallMol,Modality_E3,Modality_Ternary,Modality_Hotspot,Data_Version,Evidence_Count
        EGFR,0.8450,1,1,0,0.9230,0.7560,0.6890,0.2340,0.8120,0.6780,0.8910,0.7100,0.6500,0.7800,OT-2025.06 | STRING-v12.0 | Reactome-2024,15
        ERBB2,0.7234,2,2,0,0.8123,0.6789,0.5432,0.3456,0.7890,0.6543,0.8234,0.6200,0.5100,0.7300,OT-2025.06 | STRING-v12.0 | Reactome-2024,12"""

        return {"template": template_csv}


@app.get("/targets/{target}/modality-recommendations")
async def get_modality_recommendations(target: str):
    """
    Get modality-specific recommendations for a single target.

    This endpoint provides detailed modality analysis including:
    - PROTAC/degrader suitability
    - Small molecule druggability
    - Component scores (E3 co-expression, ternary complex, PPI hotspots)
    - Actionable recommendations
    """
    try:
        from .channels.modality_fit import get_modality_recommendations
        from .channels.ppi_proximity import ppi_network

        recommendations = get_modality_recommendations(target, ppi_network.graph)

        return {
            "target": target,
            "recommendations": recommendations,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting modality recommendations for {target}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing modality recommendations: {str(e)}"
        )


@app.get("/summary")
async def get_system_summary():
    """
    Get summary of the scoring system capabilities and data sources.
    """
    try:
        from .channels.ppi_proximity import ppi_network
        from .channels.pathway import pathway_analyzer
        from .channels.modality_fit import modality_analyzer

        return {
            "system_info": {
                "name": "VantAI Target Scoreboard",
                "version": "1.0.0",
                "description": "Modality-aware target scoring system"
            },
            "data_sources": {
                "open_targets": "GraphQL API + cache with versioning",
                "ppi_network": f"{ppi_network.graph.number_of_nodes()} nodes, {ppi_network.graph.number_of_edges()} edges",
                "pathways": f"{len(pathway_analyzer.pathways_data)} targets mapped",
                "expression_data": f"{len(modality_analyzer.expression_data)} targets",
                "ternary_data": f"{len(modality_analyzer.ternary_data)} targets"
            },
            "scoring_channels": {
                "genetics": "Open Targets association scores",
                "ppi_proximity": "Network centrality measures",
                "pathway_enrichment": "Pathway overlap and specificity",
                "modality_fit": "VantAI proprietary (E3 coexpr + ternary + hotspots)",
                "safety": "Tissue specificity and toxicity proxy"
            },
            "default_weights": {
                "genetics": 0.35,
                "ppi": 0.25,
                "pathway": 0.20,
                "safety": 0.10,
                "modality_fit": 0.10
            },
            "capabilities": [
                "Multi-target batch scoring",
                "Explainable results with evidence references",
                "Data versioning and provenance tracking",
                "Cache metadata and performance monitoring",
                "Modality-specific recommendations",
                "Configurable channel weights",
                "JSON/CSV export functionality"
            ]
        }

    except Exception as e:
        logger.error(f"Error getting system summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating system summary: {str(e)}"
        )


# Rate limiting middleware (basic implementation)
request_counts = defaultdict(int)
request_timestamps = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Basic rate limiting middleware."""
    client_ip = request.client.host

    # Clean old timestamps (older than 1 hour)
    current_time = time.time()
    request_timestamps[client_ip] = [
        ts for ts in request_timestamps[client_ip]
        if current_time - ts < 3600
    ]

    # Check rate limit (100 requests per hour per IP)
    if len(request_timestamps[client_ip]) >= 100:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 100 requests per hour."
        )

    # Add current request
    request_timestamps[client_ip].append(current_time)

    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)