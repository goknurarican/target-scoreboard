# dashboard/components/explanation_panel.py
# Separate component file for clean organization

import streamlit as st
import pandas as pd
from typing import Dict, List, Any


def render_actionable_explanation_panel(target_data: Dict, selected_target: str):
    """
    Render actionable explanation panel with clickable contributions and evidence.
    This replaces the broken HTML rendering with proper Streamlit components.
    """
    if not target_data:
        st.info("No explanation data available for this target")
        return

    # Extract explanation data - handle both new and legacy formats
    explanation = target_data.get("explanation", {})
    if not explanation:
        # Fallback to basic breakdown if no explanation object
        breakdown = target_data.get("breakdown", {})
        explanation = _build_fallback_explanation(selected_target, breakdown, target_data.get("evidence_refs", []))

    with st.container():
        st.markdown(f"### Why is {selected_target} ranked here?")
        st.caption("Click evidence badges to access external sources (PMID/database references)")

        # Channel contributions with progress bars
        contributions = explanation.get("contributions", [])
        if contributions:
            st.markdown("#### Channel Contributions")

            # Create contribution table
            for contrib in contributions:
                channel = contrib["channel"]
                weight = contrib["weight"]
                score = contrib.get("score")
                contribution = contrib["contribution"]
                available = contrib["available"]

                # Channel display names with appropriate icons
                channel_names = {
                    "genetics": "🧬 Genetics",
                    "ppi": "🕸️ PPI Network",
                    "pathway": "🔬 Pathway",
                    "safety": "⚠️ Safety",
                    "modality_fit": "💊 Modality Fit"
                }

                display_name = channel_names.get(channel, channel.title())

                # Create expandable section for each channel
                if available and score is not None:
                    with st.expander(f"{display_name}: {contribution:.3f} (Weight: {weight:.2f})", expanded=True):
                        # Progress bar showing contribution
                        max_contribution = max([c["contribution"] for c in contributions]) if contributions else 1.0
                        progress = contribution / max_contribution if max_contribution > 0 else 0
                        st.progress(progress)

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.metric("Raw Score", f"{score:.3f}")
                        with col2:
                            st.metric("Weighted", f"{contribution:.3f}")

                        # Add channel-specific interpretation
                        interpretation = _get_channel_interpretation(channel, score)
                        if interpretation:
                            st.info(interpretation)
                else:
                    # Unavailable channel - show as disabled
                    with st.expander(f"⚪ {display_name}: Not Available", expanded=False):
                        st.caption("Data not available or score is zero for this channel")

        # Evidence references section
        st.markdown("#### Supporting Evidence")
        evidence_refs = explanation.get("evidence_refs", [])

        if evidence_refs:
            # Group evidence by type for better organization
            evidence_by_type = {}
            for ref in evidence_refs:
                ref_type = ref.get("type", "other")
                if ref_type not in evidence_by_type:
                    evidence_by_type[ref_type] = []
                evidence_by_type[ref_type].append(ref)

            # Display evidence by type
            for ref_type, refs in evidence_by_type.items():
                type_labels = {
                    "literature": "📚 Literature",
                    "database": "🗄️ Databases",
                    "proprietary": "🔬 VantAI Data"
                }

                st.markdown(f"**{type_labels.get(ref_type, ref_type.title())}**")

                # Create clickable badges using columns
                cols = st.columns(min(4, len(refs)))
                for i, ref in enumerate(refs):
                    with cols[i % 4]:
                        label = ref["label"]
                        url = ref["url"]

                        if url and url != "#":
                            # External clickable link - FIXED: Use st.link_button for proper functionality
                            if st.button(label, key=f"evidence_{ref_type}_{i}", help=f"Open {url}"):
                                st.markdown(f'<script>window.open("{url}", "_blank");</script>', unsafe_allow_html=True)
                        else:
                            # Internal/unavailable - use disabled button
                            st.button(label, disabled=True, key=f"evidence_disabled_{i}_{ref_type}")
        else:
            st.info("No evidence references available")

        # Summary metrics at bottom
        total_score = explanation.get("total_weighted_score", 0)
        available_channels = sum(1 for c in contributions if c["available"])
        total_channels = len(contributions)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{total_score:.3f}")
        with col2:
            st.metric("Active Channels", f"{available_channels}/{total_channels}")
        with col3:
            confidence = "High" if available_channels >= 3 else "Medium" if available_channels >= 2 else "Low"
            st.metric("Confidence", confidence)


def _build_fallback_explanation(target: str, breakdown: Dict, evidence_refs: List[str]) -> Dict:
    """Build explanation from basic breakdown when explanation object is missing."""
    contributions = []

    # Default weights for fallback
    default_weights = {
        "genetics": 0.35,
        "ppi": 0.25,
        "pathway": 0.20,
        "safety": 0.10,
        "modality_fit": 0.10
    }

    # Build contributions from breakdown
    for channel, weight in default_weights.items():
        if channel == "ppi":
            score = breakdown.get("ppi_proximity")
        elif channel == "pathway":
            score = breakdown.get("pathway_enrichment")
        elif channel == "safety":
            score = breakdown.get("safety_off_tissue")
        elif channel == "modality_fit":
            modality_fit = breakdown.get("modality_fit", {})
            score = modality_fit.get("overall_druggability") if modality_fit else None
        else:
            score = breakdown.get(channel)

        available = score is not None and score > 0
        contribution = weight * (score if available else 0)

        contributions.append({
            "channel": channel,
            "weight": weight,
            "score": score,
            "contribution": contribution,
            "available": available
        })

    # Sort by contribution
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    # Convert evidence refs to clickable format
    clickable_evidence = []
    for ref in evidence_refs:
        if "PMID:" in ref:
            pmid = ref.split("PMID:")[1].split()[0]
            clickable_evidence.append({
                "label": f"PMID:{pmid}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                "type": "literature"
            })
        elif "OpenTargets:" in ref or "OT-" in ref:
            clickable_evidence.append({
                "label": "Open Targets",
                "url": "https://platform.opentargets.org/",
                "type": "database"
            })
        elif "STRING:" in ref:
            clickable_evidence.append({
                "label": "STRING Database",
                "url": "https://string-db.org/",
                "type": "database"
            })
        elif "Reactome:" in ref:
            clickable_evidence.append({
                "label": "Reactome",
                "url": "https://reactome.org/",
                "type": "database"
            })
        elif "VantAI:" in ref:
            clickable_evidence.append({
                "label": "VantAI Data",
                "url": "#",
                "type": "proprietary"
            })

    return {
        "target": target,
        "contributions": contributions,
        "evidence_refs": clickable_evidence,
        "total_weighted_score": sum(c["contribution"] for c in contributions)
    }


def _get_channel_interpretation(channel: str, score: float) -> str:
    """Get human-readable interpretation of channel scores."""
    interpretations = {
        "genetics": {
            (0.7, 1.0): "Strong genetic association with disease - high confidence target",
            (0.4, 0.7): "Moderate genetic evidence supports target involvement",
            (0.0, 0.4): "Limited genetic evidence - may require additional validation"
        },
        "ppi": {
            (0.6, 1.0): "Central hub in disease network - high connectivity",
            (0.3, 0.6): "Moderate network connectivity to disease genes",
            (0.0, 0.3): "Peripheral network position - may have indirect effects"
        },
        "pathway": {
            (0.6, 1.0): "Highly enriched in relevant disease pathways",
            (0.3, 0.6): "Present in some disease-relevant pathways",
            (0.0, 0.3): "Limited pathway overlap with disease mechanisms"
        },
        "safety": {
            (0.0, 0.3): "Good safety profile - low off-tissue expression",
            (0.3, 0.7): "Moderate safety concerns - some off-tissue expression",
            (0.7, 1.0): "Potential safety risks - high off-tissue expression"
        },
        "modality_fit": {
            (0.6, 1.0): "Excellent druggability - multiple modality options",
            (0.3, 0.6): "Moderate druggability - some targeting challenges",
            (0.0, 0.3): "Limited druggability - may require novel approaches"
        }
    }

    if channel not in interpretations:
        return None

    for (low, high), interpretation in interpretations[channel].items():
        if low <= score < high:
            return interpretation

    return None


def render_ranking_impact_analysis(rank_impact: List[Dict], current_weights: Dict[str, float]):
    """
    Render ranking impact analysis using native Streamlit components.
    Fixes raw HTML issues by using proper Streamlit widgets.
    """
    if not rank_impact:
        return

    # Check if weights differ from default
    default_weights = {
        "genetics": 0.35,
        "ppi": 0.25,
        "pathway": 0.20,
        "safety": 0.10,
        "modality_fit": 0.10
    }

    weights_changed = any(
        abs(current_weights.get(k, 0) - default_weights[k]) > 0.01
        for k in default_weights
    )

    if not weights_changed:
        st.info("Using default weights - no ranking changes to display")
        return

    with st.container():
        st.markdown("### Ranking Impact Analysis")
        st.caption("How current weight configuration changes rankings vs. default weights")

        # Summary of weight changes
        changes = []
        for channel, default_val in default_weights.items():
            current_val = current_weights.get(channel, default_val)
            diff = current_val - default_val
            if abs(diff) > 0.05:
                direction = "increased" if diff > 0 else "decreased"
                changes.append(f"{channel.replace('_', ' ').title()} {direction} by {abs(diff):.2f}")

        if changes:
            st.info(f"**Weight changes:** {', '.join(changes[:3])}" +
                    (f" (+{len(changes) - 3} more)" if len(changes) > 3 else ""))

        # Show top ranking changes in a grid
        significant_changes = [r for r in rank_impact if r["movement"] != "unchanged"][:9]

        if significant_changes:
            cols = st.columns(3)

            for i, impact in enumerate(significant_changes):
                with cols[i % 3]:
                    target = impact["target"]
                    rank_baseline = impact["rank_baseline"]
                    rank_current = impact["rank_current"]
                    delta = impact["delta"]
                    movement = impact["movement"]

                    # Movement styling
                    if movement == "up":
                        delta_text = f"+{delta}"
                        color = "green"
                    elif movement == "down":
                        delta_text = f"-{abs(delta)}"
                        color = "red"
                    else:
                        delta_text = "0"
                        color = "gray"

                    # Create metric card using native Streamlit
                    with st.container():
                        st.markdown(f"**{target}**")
                        st.metric(
                            "Rank Change",
                            f"{rank_baseline} → {rank_current}",
                            delta_text,
                            delta_color=color
                        )

        else:
            st.info("No significant ranking changes with current weight configuration")


def render_enhanced_results_table(target_scores: List[Dict], rank_impact: List[Dict] = None):
    """
    Render enhanced results table with ranking change indicators.
    Uses native Streamlit dataframe with proper column configuration.
    """
    if not target_scores:
        st.warning("No target scores to display")
        return

    # Sort targets by total score
    sorted_targets = sorted(target_scores, key=lambda x: x.get("total_score", 0), reverse=True)

    # Create ranking lookup
    rank_lookup = {}
    if rank_impact:
        rank_lookup = {item["target"]: item for item in rank_impact}

    # Build table data
    table_data = []
    for i, ts in enumerate(sorted_targets, 1):
        target = ts.get('target', 'Unknown')
        breakdown = ts.get("breakdown", {})
        modality_fit = breakdown.get("modality_fit", {})

        # Get ranking change info
        rank_info = rank_lookup.get(target, {})
        movement = rank_info.get("movement", "unchanged")
        delta = rank_info.get("delta", 0)

        # Movement indicator
        if movement == "up":
            rank_display = f"{i} (+{delta})"
        elif movement == "down":
            rank_display = f"{i} (-{abs(delta)})"
        else:
            rank_display = str(i)

        table_data.append({
            "Rank": rank_display,
            "Target": target,
            "Total Score": ts.get("total_score", 0),
            "Genetics": breakdown.get("genetics", 0) or 0,
            "PPI Network": breakdown.get("ppi_proximity", 0) or 0,
            "Pathway": breakdown.get("pathway_enrichment", 0) or 0,
            "Safety": breakdown.get("safety_off_tissue", 0) or 0,
            "Modality": modality_fit.get("overall_druggability", 0) if modality_fit else 0
        })

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Column configuration
    column_config = {
        "Rank": st.column_config.TextColumn("Rank", width="small",
                                            help="Current rank with movement vs default weights"),
        "Target": st.column_config.TextColumn("Target", width="medium"),
        "Total Score": st.column_config.NumberColumn("Total Score", format="%.3f", width="medium"),
        "Genetics": st.column_config.NumberColumn("Genetics", format="%.3f", width="small"),
        "PPI Network": st.column_config.NumberColumn("PPI Network", format="%.3f", width="small"),
        "Pathway": st.column_config.NumberColumn("Pathway", format="%.3f", width="small"),
        "Safety": st.column_config.NumberColumn("Safety", format="%.3f", width="small", help="Lower is better"),
        "Modality": st.column_config.NumberColumn("Modality", format="%.3f", width="small")
    }

    # Display table
    st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(500, (len(df) + 1) * 35 + 40)
    )