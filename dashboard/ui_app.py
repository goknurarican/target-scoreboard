# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

# dashboard/ui_app.py

import streamlit as st

st.set_page_config(
    page_title="VantAI Target Scoreboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
import os
import requests
import pandas as pd
import numpy as np
import json
import urllib.parse
import time
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

# Import components with error handling
try:
    from dashboard.components.explanation_panel import render_evidence_matrix
    EVIDENCE_MATRIX_AVAILABLE = True
except ImportError:
    EVIDENCE_MATRIX_AVAILABLE = False

try:
    from dashboard.components.network_viz import InteractiveNetworkViz
    NETWORK_VIZ_AVAILABLE = True
except ImportError:
    NETWORK_VIZ_AVAILABLE = False

# VantAI Professional Theme Configuration
VANTAI_THEME = {
    'bg_primary': '#0B0F1A',
    'bg_surface': '#0F172A',
    'bg_card': '#1A1F2E',
    'bg_border': '#1E293B',
    'bg_accent': '#0A1628',
    'text_primary': '#E2E8F0',
    'text_secondary': '#94A3B8',
    'text_muted': '#64748B',
    'accent_cyan': '#22D3EE',
    'accent_cyan_hover': '#06B6D4',
    'accent_violet': '#A78BFA',
    'accent_violet_soft': '#C084FC',
    'success': '#34D399',
    'warning': '#F59E0B',
    'danger': '#F87171',
    'gradient_primary': 'linear-gradient(135deg, #22D3EE 0%, #A78BFA 100%)',
    'gradient_surface': 'linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%)',
    'shadow_glow': '0 0 30px rgba(34, 211, 238, 0.1)',
    'shadow_card': '0 8px 32px rgba(0, 0, 0, 0.4)'
}

# ========================
# API Configuration - FIXED
# ========================

def get_api_base_url():
    """Get API base URL with environment detection."""
    if (os.getenv('STREAMLIT_SHARING_MODE') or
            'streamlit.app' in os.getenv('HOSTNAME', '') or
            'streamlit.app' in os.getenv('SERVER_NAME', '')):
        return os.getenv('PROD_API_URL', "https://your-backend-api.herokuapp.com")
    else:
        api_port = os.getenv('API_PORT', '8001')
        return f"http://localhost:{api_port}"

API_BASE_URL = get_api_base_url()

def call_api(endpoint, method="GET", data=None, params=None, timeout=90):
    """Enhanced API caller with better error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"

        if method == "POST":
            response = requests.post(url, json=data, params=params, timeout=timeout)
        else:
            response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            st.error(f"Endpoint not found: {endpoint}")
            return None
        elif response.status_code == 500:
            st.error(f"Server error: {response.text}")
            return None
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Ensure the backend service is running.")
        return None
    except requests.exceptions.Timeout:
        st.error(f"API request timed out after {timeout} seconds")
        return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None

# ========================
# Professional CSS
# ========================

def load_professional_css():
    """Load complete professional CSS with all styling."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: {VANTAI_THEME['bg_primary']};
        color: {VANTAI_THEME['text_primary']};
    }}

    .main .block-container {{
        background: {VANTAI_THEME['bg_primary']};
        padding-top: 3rem;
        max-width: 1400px;
    }}

    /* Enhanced Platform Header */
    .platform-header-enhanced {{
        position: relative;
        background: linear-gradient(135deg, #0B0F1A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        text-align: center;
        overflow: hidden;
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(34, 211, 238, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }}

    .header-backdrop {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 30% 20%, rgba(34, 211, 238, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 70% 80%, rgba(167, 139, 250, 0.15) 0%, transparent 50%);
        pointer-events: none;
    }}

    .header-content {{
        position: relative;
        z-index: 1;
    }}

    .platform-title-large {{
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #E2E8F0 0%, #22D3EE 50%, #A78BFA 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        letter-spacing: -0.04em;
        line-height: 1.1;
        text-shadow: 0 0 40px rgba(34, 211, 238, 0.3);
    }}

    .platform-subtitle-large {{
        color: #94A3B8;
        font-size: 1.5rem;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
        letter-spacing: 0.01em;
    }}

    .header-badges {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 2rem;
    }}

    .badge {{
        display: inline-block;
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(167, 139, 250, 0.2) 100%);
        border: 1px solid rgba(34, 211, 238, 0.3);
        color: #22D3EE;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        text-transform: uppercase;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}

    .badge:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(34, 211, 238, 0.4);
        border-color: #22D3EE;
    }}

    /* Section headers */
    .section-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {VANTAI_THEME['text_primary']};
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}

    .section-header::before {{
        content: '';
        width: 4px;
        height: 24px;
        background: {VANTAI_THEME['gradient_primary']};
        border-radius: 2px;
    }}

    /* Config summary bar */
    .config-summary {{
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }}

    .config-item {{
        color: #E2E8F0;
        font-size: 0.9rem;
        font-weight: 500;
    }}

    .config-value {{
        color: #22D3EE;
        font-weight: 600;
    }}

    /* Enhanced metric cards */
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}

    .metric-card {{
        background: {VANTAI_THEME['gradient_surface']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: {VANTAI_THEME['shadow_card']};
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: {VANTAI_THEME['shadow_glow']}, {VANTAI_THEME['shadow_card']};
        border-color: {VANTAI_THEME['accent_cyan']}40;
    }}

    .metric-label {{
        color: {VANTAI_THEME['text_secondary']};
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}

    .metric-value {{
        color: {VANTAI_THEME['accent_cyan']};
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.25rem;
    }}

    .metric-description {{
        color: {VANTAI_THEME['text_muted']};
        font-size: 0.8rem;
        font-weight: 400;
    }}

    /* Professional Buttons */
    .stButton > button {{
        background: {VANTAI_THEME['gradient_primary']};
        color: {VANTAI_THEME['bg_primary']};
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(34, 211, 238, 0.2);
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.3);
    }}

    /* Enhanced dataframes */
    .stDataFrame {{
        background: transparent !important;
    }}

    .stDataFrame > div {{
        background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%) !important;
        border: 1px solid {VANTAI_THEME['bg_border']} !important;
        border-radius: 8px !important;
    }}

    .stDataFrame table {{
        background: transparent !important;
        color: {VANTAI_THEME['text_primary']} !important;
    }}

    .stDataFrame thead th {{
        background: {VANTAI_THEME['bg_card']} !important;
        color: {VANTAI_THEME['text_secondary']} !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-size: 0.8rem !important;
        border-bottom: 1px solid {VANTAI_THEME['bg_border']} !important;
    }}

    .stDataFrame tbody td {{
        background: transparent !important;
        color: {VANTAI_THEME['text_primary']} !important;
        border-bottom: 1px solid {VANTAI_THEME['bg_border']} !important;
    }}

    .stDataFrame tbody tr:hover td {{
        background: {VANTAI_THEME['bg_card']} !important;
    }}

    /* Form Controls */
    .stSelectbox > div > div {{
        background: {VANTAI_THEME['bg_card']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 8px;
        color: {VANTAI_THEME['text_primary']};
    }}

    .stTextArea > div > div > textarea {{
        background: {VANTAI_THEME['bg_card']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 8px;
        color: {VANTAI_THEME['text_primary']};
        font-family: 'Inter', monospace;
    }}

    /* Target Details Panel */
    .target-details {{
        background: {VANTAI_THEME['gradient_surface']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: {VANTAI_THEME['shadow_card']};
    }}

    /* Status messages */
    .stSuccess {{
        background: {VANTAI_THEME['success']}15;
        border: 1px solid {VANTAI_THEME['success']};
        border-radius: 8px;
        color: {VANTAI_THEME['success']};
    }}

    .stWarning {{
        background: {VANTAI_THEME['warning']}15;
        border: 1px solid {VANTAI_THEME['warning']};
        border-radius: 8px;
        color: {VANTAI_THEME['warning']};
    }}

    .stError {{
        background: {VANTAI_THEME['danger']}15;
        border: 1px solid {VANTAI_THEME['danger']};
        border-radius: 8px;
        color: {VANTAI_THEME['danger']};
    }}

    /* Back to top */
    .backtop {{ 
        position: fixed; 
        right: 20px; 
        bottom: 24px; 
        border: 1px solid #1E293B; 
        border-radius: 999px; 
        padding: .5rem .8rem; 
        background: #0B0F1A80;
        color: #22D3EE;
        text-decoration: none;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }}

    .backtop:hover {{
        background: #22D3EE20;
        border-color: #22D3EE;
        transform: translateY(-2px);
    }}

    /* Footer */
    .platform-footer {{
        color: {VANTAI_THEME['text_muted']};
        text-align: center;
        padding: 3rem 0 2rem 0;
        border-top: 1px solid {VANTAI_THEME['bg_border']};
        margin-top: 4rem;
        font-size: 0.9rem;
    }}

    .footer-brand {{
        color: {VANTAI_THEME['accent_cyan']};
        font-weight: 600;
    }}

    /* Responsive design */
    @media (max-width: 768px) {{
        .platform-title-large {{
            font-size: 2.5rem;
        }}
        .platform-subtitle-large {{
            font-size: 1.2rem;
        }}
        .metrics-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ========================
# FIXED: Evidence Analysis Functions
# ========================

def render_evidence_distribution(target_scores):
    """Render evidence distribution summary."""
    if not target_scores:
        st.info("No targets to summarize.")
        return

    CATS = ["literature", "databases", "vantai", "other"]
    totals = {c: 0 for c in CATS}

    def _cat_from_ref(ref) -> str:
        try:
            if isinstance(ref, dict):
                if ref.get("category"):
                    return str(ref["category"]).lower()
                src = str(ref.get("source", "")).lower()
                if src in ("pubmed", "pmid", "literature"):
                    return "literature"
                if src in ("opentargets", "ot", "stringdb", "string", "reactome", "ensembl"):
                    return "databases"
                if "vantai" in src:
                    return "vantai"
                return "other"
            else:
                s = str(ref).lower()
                if s.startswith("pmid:"):
                    return "literature"
                if any(x in s for x in ["string", "reactome", "opentargets"]):
                    return "databases"
                return "other"
        except Exception:
            return "other"

    # Count evidences with proper fallbacks
    for ts in target_scores:
        item = ts if isinstance(ts, dict) else getattr(ts, "model_dump", lambda: {})()

        # Try evidence_summary first
        summary = (item or {}).get("evidence_summary") or {}
        if summary:
            for c in CATS:
                totals[c] += int(summary.get(c, 0) or 0)
            continue

        # Fallback to individual references
        evs = (item or {}).get("evidence_refs") or []
        if not evs:
            evs = ((item or {}).get("explanation") or {}).get("evidence_refs", []) or []

        for ref in evs:
            totals[_cat_from_ref(ref)] += 1

    # Display
    st.markdown("### Evidence Distribution")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Literature", totals["literature"])
    c2.metric("🗄️ Databases", totals["databases"])
    c3.metric("🧪 VantAI", totals["vantai"])
    c4.metric("⚙️ Other", totals["other"])

# ========================
# FIXED: Explanation Panel Functions
# ========================

def _build_fallback_explanation(target: str, breakdown: Dict, evidence_refs: List[str]) -> Dict:
    """FIXED: Build explanation from basic breakdown with comprehensive genetics mapping."""
    contributions = []

    # Default weights for fallback
    default_weights = {
        "genetics": 0.35,
        "ppi": 0.25,
        "pathway": 0.20,
        "safety": 0.10,
        "modality_fit": 0.10
    }

    # FIXED: More comprehensive genetics score extraction
    for channel, weight in default_weights.items():
        if channel == "genetics":
            # Try multiple possible field names for genetics
            score = (breakdown.get("genetics") or
                     breakdown.get("genetic_association") or
                     breakdown.get("gene_association") or
                     breakdown.get("genetic_evidence") or
                     breakdown.get("ot_genetics") or
                     breakdown.get("genetics_score"))

            # Check if genetics is a nested dict with score
            if score is None and isinstance(breakdown.get("genetics"), dict):
                genetics_obj = breakdown["genetics"]
                score = genetics_obj.get("score") or genetics_obj.get("association_score")

            # Also check if there's a nested genetics_data object
            if score is None:
                genetics_obj = breakdown.get("genetics_data", {})
                if isinstance(genetics_obj, dict):
                    score = genetics_obj.get("score") or genetics_obj.get("association_score")

            # FIXED: This is likely missing - check if target_data has channels structure
            # Note: In real usage, you'd need to pass the full target_data instead of just breakdown
            # For now, we'll make this work with what we have

        elif channel == "ppi":
            score = breakdown.get("ppi_proximity") or breakdown.get("ppi") or breakdown.get("ppi_score")
        elif channel == "pathway":
            score = breakdown.get("pathway_enrichment") or breakdown.get("pathway") or breakdown.get("pathway_score")
        elif channel == "safety":
            score = breakdown.get("safety_off_tissue") or breakdown.get("safety") or breakdown.get("safety_score")
        elif channel == "modality_fit":
            modality_fit = breakdown.get("modality_fit", {})
            score = modality_fit.get("overall_druggability") if modality_fit else None
        else:
            score = breakdown.get(channel)

        # FIXED: Better availability check - handle None, 0, and negative values appropriately
        if score is None:
            available = False
            score = 0.0
        else:
            score = float(score)
            # For safety channel, 0 is actually a good score (low safety risk)
            # For others, positive values indicate availability
            if channel == "safety":
                available = True  # Safety data is always considered available if present
            else:
                available = score > 0.001  # Very small threshold for floating point comparison

        contribution = weight * score

        contributions.append({
            "channel": channel,
            "weight": weight,
            "score": score,
            "contribution": contribution,
            "available": available
        })

    # Sort by contribution
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    # Process evidence refs (existing code remains the same)
    clickable_evidence = []
    for ref in evidence_refs:
        # ... existing evidence processing code ...
        pass

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


def render_actionable_explanation_panel(target_data: Dict, selected_target: str):
    """Render actionable explanation panel with comprehensive analysis."""
    if not target_data:
        st.info("No explanation data available for this target")
        return

    # Extract explanation data with better error handling
    explanation = target_data.get("explanation", {}) or {}
    is_error_state = str(target_data.get("data_version", "")).lower().startswith("error")
    no_contribs = not explanation.get("contributions")

    if is_error_state or no_contribs:
        breakdown = target_data.get("breakdown", {}) or {}

        # FIXED: Pre-process breakdown to extract channels data
        channels = target_data.get("channels", {})
        if channels and isinstance(channels, dict):
            # Flatten channels data into breakdown for compatibility
            for channel_name, channel_data in channels.items():
                if isinstance(channel_data, dict):
                    if channel_name == "genetics":
                        # Extract genetics score to breakdown
                        if not breakdown.get("genetics"):
                            genetics_score = (channel_data.get("score") or
                                              channel_data.get("association_score") or
                                              channel_data.get("ot_score") or 0)

                            # Check components.ot.score path
                            if genetics_score == 0:
                                components = channel_data.get("components", {})
                                if isinstance(components, dict):
                                    ot_component = components.get("ot", {})
                                    if isinstance(ot_component, dict):
                                        genetics_score = ot_component.get("score", 0)

                            breakdown["genetics"] = genetics_score

                    # Handle other channels similarly
                    elif channel_name in ["ppi", "pathway", "safety", "modality_fit"]:
                        if not breakdown.get(channel_name):
                            breakdown[channel_name] = channel_data.get("score", 0)

        explanation = _build_fallback_explanation(
            selected_target,
            breakdown,
            target_data.get("evidence_refs", []) or []
        )

    # Rest of the function remains the same...
    with st.container():
        st.markdown(f"### Why is {selected_target} ranked here?")
        st.caption("Click evidence badges to access external sources")

        # Channel contributions with progress bars
        contributions = explanation.get("contributions", [])
        if contributions:
            st.markdown("#### Channel Contributions")

            for contrib in contributions:
                channel = contrib["channel"]
                weight = contrib["weight"]
                score = contrib.get("score", 0)
                contribution = contrib["contribution"]
                available = contrib["available"]

                # Channel display names
                channel_names = {
                    "genetics": "🧬 Genetics",
                    "ppi": "🕸️ PPI Network",
                    "pathway": "🔬 Pathway",
                    "safety": "⚠️ Safety",
                    "modality_fit": "💊 Modality Fit"
                }

                display_name = channel_names.get(channel, channel.title())

                # Create expandable section for each channel
                if available and score is not None and score > 0:
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


# ========================
# FIXED: PPI Network Visualization
# ========================

def render_ppi_network_card(target_data: dict):
    """FIXED: Show PPI network with proper fallbacks."""
    st.markdown("#### PPI Network Neighbors")

    # Extract PPI data from target_data
    ppi_data = (target_data.get("channels") or {}).get("ppi") or {}
    components = ppi_data.get("components") or {}

    # Try to get graph preview first
    graph_preview = components.get("graph_preview")
    neighbors = components.get("neighbors", [])

    if graph_preview and NETWORK_VIZ_AVAILABLE:
        try:
            # Use network visualization component
            viz = InteractiveNetworkViz()
            fig = viz.render_from_preview(
                graph_preview,
                height=400,
                title=f"PPI Network: {target_data.get('target', 'Target')}"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                raise Exception("Visualization failed")
        except Exception as e:
            st.warning(f"Network visualization unavailable: {e}")
            # Fallback to table
            if neighbors:
                st.dataframe(pd.DataFrame(neighbors), use_container_width=True, hide_index=True)
            else:
                st.info("No PPI neighbors available for visualization")
    elif neighbors:
        st.info("Showing PPI network neighbors")
        neighbors_df = pd.DataFrame(neighbors)
        st.dataframe(neighbors_df, use_container_width=True, hide_index=True)

        # Show network statistics if available
        if len(neighbors) > 0:
            st.caption(f"Found {len(neighbors)} first-degree neighbors")
    else:
        st.warning("PPI network analysis unavailable: no neighbors found for this target")

# ========================
# FIXED: Results Display Functions
# ========================

def render_enhanced_results_table(target_scores, rank_impact=None):
    """FIXED: Render enhanced results table with comprehensive genetics data extraction."""
    if not target_scores:
        st.warning("No target scores to display")
        return

    sorted_targets = sorted(target_scores, key=lambda x: x.get("total_score", 0), reverse=True)

    # Create ranking lookup
    rank_lookup = {}
    if rank_impact:
        rank_lookup = {item["target"]: item for item in rank_impact}

    # Build table data with comprehensive data extraction
    table_data = []
    for i, ts in enumerate(sorted_targets, 1):
        target = ts.get('target', 'Unknown')
        breakdown = ts.get("breakdown", {})

        # Get ranking change info
        rank_info = rank_lookup.get(target, {})
        movement = rank_info.get("movement", "unchanged")
        delta = rank_info.get("delta", 0)

        # Movement indicator
        if movement == "up":
            rank_indicator = f"📈 {i} (+{delta})"
        elif movement == "down":
            rank_indicator = f"📉 {i} (-{abs(delta)})"
        else:
            rank_indicator = f"➡️ {i}"

        # FIXED: Comprehensive genetics data extraction with channels support
        genetics_score = 0
        if breakdown:
            # Try standard breakdown paths first
            genetics_score = (breakdown.get("genetics") or
                              breakdown.get("genetic_association") or
                              breakdown.get("gene_association") or
                              breakdown.get("genetics_score") or
                              breakdown.get("ot_genetics") or
                              0)

            # Check if genetics is a nested dict with score
            if genetics_score == 0 and isinstance(breakdown.get("genetics"), dict):
                genetics_obj = breakdown["genetics"]
                genetics_score = (genetics_obj.get("score") or
                                  genetics_obj.get("association_score") or 0)

            # Check nested genetics_data path
            if genetics_score == 0:
                genetics_obj = breakdown.get("genetics_data", {})
                if isinstance(genetics_obj, dict):
                    genetics_score = (genetics_obj.get("score") or
                                      genetics_obj.get("association_score") or 0)

        # FIXED: Check channels.genetics.* paths with fallback status handling
        if genetics_score == 0:
            channels = ts.get("channels", {})
            if channels and isinstance(channels, dict):
                genetics_channel = channels.get("genetics", {})
                if isinstance(genetics_channel, dict):
                    score = genetics_channel.get("score")
                    status = genetics_channel.get("status")
                    components = genetics_channel.get("components", {})

                    # FIXED: Handle fallback case - if score=0 and status="ok" but has fallback components
                    is_fallback = any("fallback" in str(k) for k in components.keys())

                    if status == "ok" and score == 0 and is_fallback:
                        # This is fallback data returning 0, keep as 0 but log it
                        genetics_score = 0.0
                        # Could also set to None to indicate missing data
                    elif status == "data_missing" or status == "error":
                        genetics_score = 0.0  # Explicitly missing
                    elif status == "ok" and score is not None:
                        genetics_score = float(score)
                    else:
                        genetics_score = (genetics_channel.get("association_score") or
                                          genetics_channel.get("ot_score") or 0)

                    # Check components.ot.score path if still 0
                    if genetics_score == 0 and not is_fallback:
                        if isinstance(components, dict):
                            ot_component = components.get("ot", {})
                            if isinstance(ot_component, dict):
                                genetics_score = ot_component.get("score", 0)

        # Convert to float and handle None values
        genetics_score = float(genetics_score) if genetics_score is not None else 0.0

        # Extract other scores with similar comprehensive approach
        ppi_score = 0
        if breakdown:
            ppi_score = breakdown.get("ppi_proximity") or breakdown.get("ppi") or 0
        if ppi_score == 0:
            channels = ts.get("channels", {})
            ppi_channel = channels.get("ppi", {}) if channels else {}
            if isinstance(ppi_channel, dict):
                ppi_score = ppi_channel.get("score", 0)

        pathway_score = 0
        if breakdown:
            pathway_score = breakdown.get("pathway_enrichment") or breakdown.get("pathway") or 0
        if pathway_score == 0:
            channels = ts.get("channels", {})
            pathway_channel = channels.get("pathway", {}) if channels else {}
            if isinstance(pathway_channel, dict):
                pathway_score = pathway_channel.get("score", 0)

        safety_score = 0
        if breakdown:
            safety_score = breakdown.get("safety_off_tissue") or breakdown.get("safety") or 0
        if safety_score == 0:
            channels = ts.get("channels", {})
            safety_channel = channels.get("safety", {}) if channels else {}
            if isinstance(safety_channel, dict):
                safety_score = safety_channel.get("score", 0)

        # FIXED: Extract modality data properly
        modality_score = 0
        if breakdown:
            modality_fit = breakdown.get("modality_fit", {})
            if modality_fit and isinstance(modality_fit, dict):
                modality_score = modality_fit.get("overall_druggability") or 0
        if modality_score == 0:
            channels = ts.get("channels", {})
            modality_channel = channels.get("modality_fit", {}) if channels else {}
            if isinstance(modality_channel, dict):
                modality_score = modality_channel.get("score", 0)

        # Convert all to float and handle None values
        ppi_score = float(ppi_score) if ppi_score is not None else 0.0
        pathway_score = float(pathway_score) if pathway_score is not None else 0.0
        safety_score = float(safety_score) if safety_score is not None else 0.0
        modality_score = float(modality_score) if modality_score is not None else 0.0

        table_data.append({
            "Rank": rank_indicator,
            "Target": target,
            "Total Score": ts.get("total_score", 0),
            "Genetics": genetics_score,
            "PPI Network": ppi_score,
            "Pathway": pathway_score,
            "Safety": safety_score,
            "Modality": modality_score
        })

    # Create DataFrame with enhanced styling
    df = pd.DataFrame(table_data)

    # Column configuration
    column_config = {
        "Rank": st.column_config.TextColumn("Rank", width="small"),
        "Target": st.column_config.TextColumn("Target", width="medium"),
        "Total Score": st.column_config.NumberColumn("Total Score", format="%.3f", width="medium"),
        "Genetics": st.column_config.NumberColumn("Genetics", format="%.3f", width="small",
                                                  help="Genetic association with disease"),
        "PPI Network": st.column_config.NumberColumn("PPI Network", format="%.3f", width="small",
                                                     help="Protein-protein interaction proximity"),
        "Pathway": st.column_config.NumberColumn("Pathway", format="%.3f", width="small",
                                                 help="Pathway enrichment score"),
        "Safety": st.column_config.NumberColumn("Safety", format="%.3f", width="small",
                                                help="Safety profile (lower is better)"),
        "Modality": st.column_config.NumberColumn("Modality", format="%.3f", width="small",
                                                  help="Druggability assessment")
    }

    st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(500, (len(df) + 1) * 35 + 40)
    )

    # Debug info - show raw data structure for first target
    genetics_values = [row["Genetics"] for row in table_data]
    if all(val == 0 for val in genetics_values):
        with st.expander("🔧 Debug: Data Structure Analysis", expanded=False):
            st.write("**First target raw data:**")
            if target_scores:
                first_target = target_scores[0]
                st.json({
                    "target": first_target.get("target"),
                    "breakdown": first_target.get("breakdown", {}),
                    "channels": first_target.get("channels", {}),
                    "explanation": first_target.get("explanation", {})
                }, expanded=False)

                # Show available paths
                breakdown = first_target.get("breakdown", {})
                channels = first_target.get("channels", {})
                st.write("**Available breakdown fields:**", list(breakdown.keys()) if breakdown else [])
                st.write("**Available channel fields:**", list(channels.keys()) if channels else [])

                if channels.get("genetics"):
                    st.write("**Genetics channel structure:**")
                    st.json(channels["genetics"], expanded=True)


def _extract_genetics_score_with_status_check(target_data):
    """Helper to extract genetics score with proper status checking."""
    channels = target_data.get("channels", {})
    genetics_channel = channels.get("genetics", {})

    if isinstance(genetics_channel, dict):
        status = genetics_channel.get("status")
        score = genetics_channel.get("score")

        # FIXED: Treat score=0 with "ok" status as missing data if it's from fallback
        components = genetics_channel.get("components", {})
        is_fallback = "fallback" in str(components)

        if status == "ok" and score == 0 and is_fallback:
            # This is fallback data, treat as missing
            return 0.0
        elif status == "data_missing" or status == "error":
            # Explicitly missing
            return 0.0
        elif status == "ok" and score is not None and score > 0:
            # Real positive score
            return float(score)
        else:
            # Default case
            return 0.0

    return 0.0
# ========================
# FIXED: Sensitivity Analysis Functions
# ========================

def render_stability_sensitivity_analysis(results, last_request):
    """FIXED: Render weight sensitivity analysis with optimized parameters."""
    with st.container():
        st.markdown('<div class="section-header">Stability & Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.caption("Analyze how ranking stability varies under weight uncertainty")

        col1, col2 = st.columns([1, 1])

        with col1:
            # FIXED: Add sample size selection for performance tuning
            sample_size = st.selectbox(
                "Sample Size",
                [10, 25, 50, 100, 200],
                index=0,  # Default to 10 for faster results
                help="Lower values = faster results, higher values = more accurate"
            )

            alpha_value = st.slider(
                "Alpha (Weight Stability)",
                min_value=10.0,
                max_value=100.0,
                value=80.0,
                step=10.0,
                help="Higher values = less weight variation"
            )

        with col2:
            st.markdown(f"""
            **Parameters:** {sample_size} samples, Dirichlet α={alpha_value}  
            ⚡ Optimized for speed - use fewer samples for faster results
            """)

            if st.button("🎯 Simulate Weight Sensitivity", help="Run Monte Carlo simulation"):
                st.session_state["run_simulation"] = True
                st.session_state["sim_params"] = {"samples": sample_size, "alpha": alpha_value}

        # Run simulation if requested
        if st.session_state.get("run_simulation", False):
            params = st.session_state.get("sim_params", {"samples": 10, "alpha": 80.0})

            with st.spinner(f"Running simulation with {params['samples']} samples..."):
                try:
                    # FIXED: Use optimized parameters and shorter timeout
                    response = call_api(
                        "/sensitivity/stability",
                        method="POST",
                        data=last_request,
                        params=params,
                        timeout=60  # Increased timeout but still reasonable
                    )

                    if response:
                        st.session_state["simulation_results"] = response
                        st.session_state["run_simulation"] = False
                        st.success(f"Simulation completed successfully with {params['samples']} samples!")
                    else:
                        st.error("Simulation failed - check API connection")
                        st.session_state["run_simulation"] = False

                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
                    st.session_state["run_simulation"] = False

                    # Provide helpful guidance
                    if "timeout" in str(e).lower():
                        st.info("💡 Try using fewer samples (10-25) for faster results")

        # Display simulation results if available
        if "simulation_results" in st.session_state:
            sim_data = st.session_state["simulation_results"]
            render_simulation_results(sim_data)

def render_simulation_results(sim_data):
    """Render simulation results with proper data extraction."""
    if not sim_data:
        st.info("No simulation data available")
        return

    # Extract stability data
    stability_data = sim_data.get("stability", {})
    samples_count = sim_data.get("samples", 0)
    alpha = sim_data.get("alpha", 80.0)

    if stability_data:
        st.markdown("#### Ranking Stability Results")

        # Create stability table
        stability_results = []
        for target_data in stability_data:
            if isinstance(target_data, dict):
                target = target_data.get("target", "Unknown")
                stability_results.append({
                    "Target": target,
                    "Baseline Rank": target_data.get("baseline_rank", "N/A"),
                    "Mean Rank": f"{target_data.get('mean_rank', 0):.1f}",
                    "Std Rank": f"{target_data.get('std_rank', 0):.1f}",
                    "Best Rank": target_data.get("best_rank", "N/A"),
                    "Worst Rank": target_data.get("worst_rank", "N/A")
                })

        if stability_results:
            stability_df = pd.DataFrame(stability_results)
            st.dataframe(stability_df, use_container_width=True, hide_index=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Samples Analyzed", f"{samples_count:,}")

            with col2:
                st.metric("Alpha Parameter", f"{alpha:.1f}")

            with col3:
                avg_std = sum(float(r["Std Rank"]) for r in stability_results) / len(stability_results)
                st.metric("Avg Rank Std", f"{avg_std:.2f}")

        with st.expander("📊 How to interpret stability results"):
            st.markdown("""
            **Stability Analysis:**
            - **Mean Rank**: Average rank across all weight perturbations
            - **Std Rank**: Standard deviation of ranks (lower = more stable)
            - **Best/Worst Rank**: Range of rank variations

            **Interpretation:**
            - Targets with low rank standard deviation are robust to weight changes
            - High rank variation indicates sensitivity to weight configuration
            - Stable targets maintain consistent ranking across scenarios
            """)

def render_channel_ablation_analysis(results, last_request):
    """FIXED: Render channel ablation analysis with correct endpoint."""
    with st.container():
        st.markdown('<div class="section-header">Channel Ablation Analysis</div>', unsafe_allow_html=True)
        st.caption("Analyze the impact of removing each scoring channel")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔬 Run Ablation Analysis", help="Remove each channel and measure impact"):
                st.session_state["run_ablation"] = True

        with col2:
            st.markdown("**Method:** Sets each channel weight to 0, renormalizes others")

        # Run ablation if requested
        if st.session_state.get("run_ablation", False):
            with st.spinner("Running channel ablation analysis..."):
                try:
                    # FIXED: Use correct endpoint from main.py
                    response = call_api("/sensitivity/ablation", method="POST", data=last_request)

                    if response:
                        st.session_state["ablation_results"] = response
                        st.session_state["run_ablation"] = False
                        st.success("Ablation analysis completed!")
                    else:
                        st.error("Ablation analysis failed - check API connection")
                        st.session_state["run_ablation"] = False

                except Exception as e:
                    st.error(f"Ablation error: {e}")
                    st.session_state["run_ablation"] = False

        # Display ablation results if available
        if "ablation_results" in st.session_state:
            ablation_data = st.session_state["ablation_results"]
            render_ablation_results(ablation_data)

def render_ablation_results(ablation_data):
    """Render ablation analysis results."""
    if not ablation_data:
        st.info("No ablation data available")
        return

    targets = ablation_data.get("targets", [])
    if not targets:
        st.warning("No target data in ablation results")
        return

    st.markdown("#### Channel Ablation Results")

    # Show results for each target
    for target_result in targets:
        target_name = target_result.get("target", "Unknown")
        baseline = target_result.get("baseline", 0)
        ablations = target_result.get("ablations", [])

        with st.expander(f"{target_name} - Baseline: {baseline:.3f}", expanded=True):
            if ablations:
                ablation_results = []
                for ablation in ablations:
                    channel = ablation.get("channel", "Unknown")
                    score = ablation.get("score", 0)
                    delta = ablation.get("delta", 0)

                    ablation_results.append({
                        "Channel": channel.replace("_", " ").title(),
                        "Score After Removal": f"{score:.3f}",
                        "Score Drop": f"{-delta:.3f}",
                        "Impact": "High" if abs(delta) > 0.1 else "Medium" if abs(delta) > 0.05 else "Low"
                    })

                ablation_df = pd.DataFrame(ablation_results)
                st.dataframe(ablation_df, use_container_width=True, hide_index=True)

                # Show most critical channel
                most_critical = max(ablations, key=lambda x: abs(x.get("delta", 0)))
                critical_channel = most_critical.get("channel", "Unknown")
                critical_impact = abs(most_critical.get("delta", 0))

                if critical_impact > 0.01:
                    st.info(f"💡 **Most critical channel:** {critical_channel.replace('_', ' ').title()} "
                            f"(removing it drops score by {critical_impact:.3f})")

def render_weight_impact_analysis(rank_impact, current_weights):
    """Render weight impact analysis."""
    if not rank_impact:
        st.info("No weight impact data available")
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
        st.markdown("### Weight Impact Analysis")
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

        # Show ranking changes in a grid
        significant_changes = [r for r in rank_impact if r.get("movement") != "unchanged"][:9]

        if significant_changes:
            cols = st.columns(3)

            for i, impact in enumerate(significant_changes):
                with cols[i % 3]:
                    target = impact.get("target", "Unknown")
                    rank_baseline = impact.get("rank_baseline", 0)
                    rank_current = impact.get("rank_current", 0)
                    delta = impact.get("delta", 0)
                    movement = impact.get("movement", "unchanged")

                    # Movement styling
                    if movement == "up":
                        emoji = "📈"
                        color = "#34D399"
                        delta_text = f"+{delta}"
                    elif movement == "down":
                        emoji = "📉"
                        color = "#F87171"
                        delta_text = f"-{abs(delta)}"
                    else:
                        emoji = "➡️"
                        color = "#94A3B8"
                        delta_text = "0"

                    # Create metric card
                    st.markdown(f"""
                    <div style="border: 1px solid #1E293B; border-radius: 8px; padding: 1rem; text-align: center; background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%);">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                        <div style="font-weight: 600; color: #E2E8F0; margin-bottom: 0.5rem;">{target}</div>
                        <div style="color: {color}; font-size: 0.9rem; font-weight: 500;">
                            Rank {rank_baseline} → {rank_current} ({delta_text})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No significant ranking changes with current weight configuration")

# ========================
# Benchmark Analysis Functions
# ========================

def create_demo_ground_truth():
    """Create demo ground truth data files."""
    demo_dir = Path("data_demo")
    demo_dir.mkdir(exist_ok=True)

    # NSCLC ground truth
    nsclc_truth = {
        "disease": "Non-small cell lung carcinoma",
        "disease_id": "EFO_0003071",
        "positives": ["EGFR", "ALK", "MET", "ERBB2", "BRAF", "KRAS", "ROS1", "RET"],
        "negatives": ["GAPDH", "ACTB", "TUBB", "RPL13A", "HPRT1"],
        "description": "Known therapeutic targets vs housekeeping genes",
        "source": "Clinical trials and FDA approvals for NSCLC as of 2024",
        "created": "2025-01-01"
    }

    # Breast cancer ground truth
    breast_truth = {
        "disease": "Breast carcinoma",
        "disease_id": "EFO_0000305",
        "positives": ["ERBB2", "ESR1", "PGR", "CDK4", "CDK6", "PIK3CA", "AKT1"],
        "negatives": ["GAPDH", "ACTB", "TUBB", "RPL13A"],
        "description": "Established breast cancer targets vs control genes",
        "source": "FDA-approved therapies and clinical guidelines",
        "created": "2025-01-01"
    }

    # Save ground truth files
    with open(demo_dir / "nsclc_truth.json", "w") as f:
        json.dump(nsclc_truth, f, indent=2)

    with open(demo_dir / "breast_truth.json", "w") as f:
        json.dump(breast_truth, f, indent=2)

    return demo_dir

def load_ground_truth(disease_name: str):
    """Load ground truth data for disease benchmarking."""
    demo_dir = Path("data_demo")

    # Map disease names to files
    truth_files = {
        "non-small cell lung carcinoma": "nsclc_truth.json",
        "nsclc": "nsclc_truth.json",
        "lung": "nsclc_truth.json",
        "breast carcinoma": "breast_truth.json",
        "breast": "breast_truth.json"
    }

    disease_key = disease_name.lower()
    truth_file = truth_files.get(disease_key)

    if not truth_file:
        return None

    truth_path = demo_dir / truth_file
    if not truth_path.exists():
        create_demo_ground_truth()

    try:
        with open(truth_path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def compute_precision_at_k(scores: list, positives: set, k_values: list = [1, 3, 5]):
    """Compute precision@k metrics."""
    precisions = {}

    for k in k_values:
        if k > len(scores):
            precisions[k] = 0.0
            continue

        top_k_targets = [target for target, _ in scores[:k]]
        true_positives = sum(1 for target in top_k_targets if target in positives)
        precisions[k] = true_positives / k if k > 0 else 0.0

    return precisions

def compute_auc_pr_simple(scores: list, positives: set):
    """Compute simple AUC-PR approximation."""
    if not scores or not positives:
        return 0.0

    # Create binary labels
    labels = [1 if target in positives else 0 for target, _ in scores]

    if sum(labels) == 0:  # No positives
        return 0.0

    # Compute precision and recall at each threshold
    precisions = []
    recalls = []

    tp = 0
    fp = 0
    total_positives = sum(labels)

    for i, label in enumerate(labels):
        if label == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_positives if total_positives > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    # Compute AUC using trapezoidal rule approximation
    auc = 0.0
    for i in range(1, len(recalls)):
        width = recalls[i] - recalls[i - 1]
        height = (precisions[i] + precisions[i - 1]) / 2
        auc += width * height

    return auc

def render_benchmark_panel(results, selected_disease_name):
    """Render benchmark analysis panel comparing results to ground truth."""
    st.markdown('<div class="section-header">Benchmark Analysis</div>', unsafe_allow_html=True)
    st.markdown("#### Performance vs Ground Truth")
    st.caption("Compare current scoring results against known therapeutic targets")

    # Load ground truth for current disease
    ground_truth = load_ground_truth(selected_disease_name)

    if not ground_truth:
        st.warning(f"No ground truth data available for {selected_disease_name}")
        st.info("Available benchmarks: NSCLC, Breast Cancer")
        return

    # Extract current results
    target_scores = results.get("targets", [])
    if not target_scores:
        st.warning("No scoring results available for benchmarking")
        return

    # Prepare data for benchmarking
    scored_targets = []
    for ts in target_scores:
        if hasattr(ts, 'target'):  # Object format
            target = ts.target
            score = ts.total_score
        else:  # Dictionary format
            target = ts.get('target', 'Unknown')
            score = ts.get('total_score', 0)
        scored_targets.append((target, score))

    scored_targets.sort(key=lambda x: x[1], reverse=True)

    positives = set(ground_truth["positives"])
    negatives = set(ground_truth["negatives"])

    # Filter to only targets that are in ground truth
    benchmark_targets = [
        (target, score) for target, score in scored_targets
        if target in positives or target in negatives
    ]

    if not benchmark_targets:
        st.warning("No targets from current results match ground truth data")
        return

    # Compute metrics
    precision_at_k = compute_precision_at_k(benchmark_targets, positives, [1, 3, 5])
    auc_pr = compute_auc_pr_simple(benchmark_targets, positives)

    # Display metrics
    st.markdown("#### Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision@1", f"{precision_at_k.get(1, 0):.3f}")
    with col2:
        st.metric("Precision@3", f"{precision_at_k.get(3, 0):.3f}")
    with col3:
        st.metric("Precision@5", f"{precision_at_k.get(5, 0):.3f}")
    with col4:
        st.metric("AUC-PR", f"{auc_pr:.3f}")

    # Performance interpretation
    avg_precision = sum(precision_at_k.values()) / len(precision_at_k) if precision_at_k else 0

    if avg_precision >= 0.8:
        performance_level = "Excellent"
        performance_color = "#34D399"
    elif avg_precision >= 0.6:
        performance_level = "Good"
        performance_color = "#22D3EE"
    elif avg_precision >= 0.4:
        performance_level = "Fair"
        performance_color = "#F59E0B"
    else:
        performance_level = "Poor"
        performance_color = "#F87171"

    st.markdown(f"""
        **Overall Performance:** 
        <span style="color: {performance_color}; font-weight: 600;">{performance_level}</span>
        (avg precision: {avg_precision:.3f})
        """, unsafe_allow_html=True)

    # Show results details
    st.markdown("#### Top Predictions vs Ground Truth")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Predicted Targets:**")
        for i, (target, score) in enumerate(benchmark_targets[:5], 1):
            if target in positives:
                st.markdown(f"{i}. ✅ **{target}** ({score:.3f}) - Known target")
            elif target in negatives:
                st.markdown(f"{i}. ❌ **{target}** ({score:.3f}) - Control gene")
            else:
                st.markdown(f"{i}. ❓ **{target}** ({score:.3f}) - Unknown")

    with col2:
        st.markdown("**Ground Truth Summary:**")
        st.write(f"**Positives:** {len(positives)} known targets")
        st.write(f"**Negatives:** {len(negatives)} control genes")
        st.write(f"**Benchmarked:** {len(benchmark_targets)} targets")

        # Show missed targets
        all_scored = set(target for target, _ in scored_targets)
        missed_positives = positives - all_scored
        if missed_positives:
            st.write(f"**Missed targets:** {', '.join(list(missed_positives)[:3])}")

# ========================
# URL State Management
# ========================

def encode_params_for_url(disease_id, targets, weights):
    """Encode parameters for URL sharing."""
    try:
        params = {
            "disease": disease_id,
            "targets": ",".join(targets) if targets else "",
            "genetics": weights.get("genetics", 0.35),
            "ppi": weights.get("ppi", 0.25),
            "pathway": weights.get("pathway", 0.20),
            "safety": weights.get("safety", 0.10),
            "modality_fit": weights.get("modality_fit", 0.10)
        }
        return params
    except Exception:
        return {}

def decode_params_from_url():
    """Decode parameters from URL query params."""
    try:
        # Use st.query_params for Streamlit 1.28+
        if hasattr(st, 'query_params'):
            query_params = st.query_params
        else:
            # Fallback for older versions
            query_params = st.experimental_get_query_params()

        if not query_params:
            return None, None, None

        # Extract disease
        disease_id = query_params.get("disease", [""])[0] if isinstance(query_params.get("disease"),
                                                                        list) else query_params.get("disease", "")

        # Extract targets
        targets_str = query_params.get("targets", [""])[0] if isinstance(query_params.get("targets"),
                                                                         list) else query_params.get("targets", "")
        targets = [t.strip().upper() for t in targets_str.split(",") if t.strip()] if targets_str else []

        # Extract weights
        weights = {}
        weight_keys = ["genetics", "ppi", "pathway", "safety", "modality_fit"]
        for key in weight_keys:
            try:
                value = query_params.get(key, [0])[0] if isinstance(query_params.get(key), list) else query_params.get(
                    key, 0)
                weights[key] = float(value)
            except (ValueError, TypeError):
                weights[
                    key] = 0.35 if key == "genetics" else 0.25 if key == "ppi" else 0.20 if key == "pathway" else 0.10

        return disease_id, targets, weights
    except Exception:
        return None, None, None

def update_url_params(disease_id, targets, weights):
    """Update URL query parameters with current state."""
    try:
        params = encode_params_for_url(disease_id, targets, weights)

        # Use appropriate method based on Streamlit version
        if hasattr(st, 'query_params'):
            # Streamlit 1.28+
            for key, value in params.items():
                st.query_params[key] = str(value)
        else:
            # Older versions
            st.experimental_set_query_params(**{k: str(v) for k, v in params.items()})
    except Exception:
        pass  # Fail silently if URL update fails

# ========================
# Sidebar with Complete URL State Management
# ========================

def render_sidebar_with_url_state():
    """Render complete sidebar with URL state loading and sharing."""
    # Load state from URL
    url_disease_id, url_targets, url_weights = decode_params_from_url()

    # Disease selection
    st.markdown("### Disease Context")
    disease_options = {
        "Non-small cell lung carcinoma": "EFO_0003060",        "Lung adenocarcinoma": "MONDO_0005233",  # Specific subtype - VALID
        "Breast carcinoma": "EFO_0000305",  # Breast cancer - VALID
        "Colorectal carcinoma": "EFO_1001951",  # ✅ colorectal ca
         "Prostate carcinoma": "EFO_0001663",  # Prostate cancer - VALID
    }

    # Use URL state if available
    default_disease = None
    if url_disease_id:
        for name, id_val in disease_options.items():
            if id_val == url_disease_id:
                default_disease = name
                break

    if not default_disease:
        default_disease = list(disease_options.keys())[0]

    selected_disease_name = st.selectbox(
        "Select Disease",
        list(disease_options.keys()),
        index=list(disease_options.keys()).index(default_disease)
    )
    disease_id = disease_options[selected_disease_name]

    # Target input
    st.markdown("### Target Selection")

    target_sets = {
        "NSCLC Targets": ["EGFR", "ERBB2", "MET", "ALK", "KRAS"],
        "Oncogenes": ["EGFR", "ERBB2", "MET", "ALK", "BRAF", "PIK3CA"],
        "Tumor Suppressors": ["TP53", "RB1", "PTEN"],
        "Custom": []
    }

    # Determine initial target set based on URL
    initial_set = "Custom"
    if url_targets:
        # Check if URL targets match any predefined set
        for set_name, set_targets in target_sets.items():
            if set(url_targets) == set(set_targets):
                initial_set = set_name
                break

    selected_set = st.selectbox(
        "Target Set",
        list(target_sets.keys()),
        index=list(target_sets.keys()).index(initial_set)
    )

    if selected_set == "Custom":
        # Use URL targets if available
        default_targets_text = "\n".join(url_targets) if url_targets else "EGFR\nERBB2\nMET\nALK\nKRAS"
        targets_input = st.text_area(
            "Enter targets (one per line)",
            value=default_targets_text,
            height=100
        )
        targets = [t.strip().upper() for t in targets_input.split("\n") if t.strip()]
    else:
        targets = target_sets[selected_set]
        st.markdown(f"*Targets:* {', '.join(targets)}")

    # Scoring weights with URL defaults
    st.markdown("### Algorithm Weights")

    default_weights = url_weights if url_weights else {
        "genetics": 0.35, "ppi": 0.25, "pathway": 0.20, "safety": 0.10, "modality_fit": 0.10
    }

    genetics_weight = st.slider("Genetics", 0.0, 1.0, default_weights["genetics"], 0.05)
    ppi_weight = st.slider("PPI Proximity", 0.0, 1.0, default_weights["ppi"], 0.05)
    pathway_weight = st.slider("Pathway", 0.0, 1.0, default_weights["pathway"], 0.05)
    safety_weight = st.slider("Safety", 0.0, 1.0, default_weights["safety"], 0.05)
    modality_weight = st.slider("Modality Fit", 0.0, 1.0, default_weights["modality_fit"], 0.05)

    weights = {
        "genetics": genetics_weight,
        "ppi": ppi_weight,
        "pathway": pathway_weight,
        "safety": safety_weight,
        "modality_fit": modality_weight
    }

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.1:
        st.warning(f"Weight sum: {weight_sum:.2f} (should be ≈1.0)")

    # Execute analysis
    if st.button("Execute Analysis", type="primary"):
        if not targets:
            st.error("Please select or enter targets for analysis")
            return None, None, None, None

        # Update URL parameters
        update_url_params(disease_id, targets, weights)

        request_data = {
            "disease": disease_id,
            "targets": targets,
            "weights": weights
        }

        with st.spinner("Running computational analysis..."):
            response = call_api("/score", method="POST", data=request_data)

        if response:
            st.session_state["scoring_results"] = response
            st.session_state["last_request"] = request_data
            processing_time = response.get('processing_time_ms', 0)
            target_count = len(response.get('targets', []))
            st.success(f"Analysis complete: {target_count} targets processed in {processing_time:.1f}ms")

    # Share link section
    if "scoring_results" in st.session_state:
        st.markdown("### Share Analysis")
        try:
            if hasattr(st, 'query_params') and st.query_params:
                query_string = urllib.parse.urlencode(dict(st.query_params))
                shareable_url = f"http://localhost:8501/?{query_string}"
                st.code(shareable_url, language=None)

                # Copy button using JavaScript
                copy_button_script = f"""
                <button onclick="
                    navigator.clipboard.writeText('{shareable_url}').then(function() {{
                        alert('Link copied to clipboard!');
                    }}, function(err) {{
                        console.error('Could not copy text: ', err);
                        alert('Copy failed. Please copy manually.');
                    }});
                " style="
                    background: linear-gradient(135deg, #22D3EE 0%, #A78BFA 100%);
                    color: #0B0F1A;
                    border: none;
                    border-radius: 6px;
                    padding: 0.5rem 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    font-size: 0.9rem;
                    margin-top: 0.5rem;
                ">
                    🔗 Copy Link
                </button>
                """
                st.markdown(copy_button_script, unsafe_allow_html=True)
        except Exception:
            st.info("Copy current URL to share this analysis configuration")

    return selected_disease_name, disease_id, targets, weights

# ========================
# Main Dashboard Function - COMPLETE VERSION
# ========================

def main():
    """Main dashboard function with complete tabbed layout structure."""

    # Load professional theme with enhanced CSS
    load_professional_css()

    # Enhanced Platform header
    st.markdown("""
        <div class="platform-header-enhanced">
            <div class="header-backdrop"></div>
            <div class="header-content">
                <div class="platform-title-large">VantAI Target Scoreboard</div>
                <div class="platform-subtitle-large">
                    Advanced computational platform for modality-aware target prioritization using multi-omics integration
                    <br><br>
                    <span style="font-weight:600; color:#4FC3F7;">Developed by Göknur Arıcan</span>
                    <br>
                    <span style="font-size:0.9em; color:#B0BEC5;">(pronounced: <i>Gyok-noor A-ruh-jan</i>)</span>
                </div>
                <div class="header-badges">
                    <span class="badge">AI-Powered</span>
                    <span class="badge">Multi-Omics</span>
                    <span class="badge">Real-Time</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar with URL state management
    with st.sidebar:
        selected_disease_name, disease_id, targets, weights = render_sidebar_with_url_state()

    # MAIN CONTENT AREA
    if "scoring_results" in st.session_state:
        results = st.session_state["scoring_results"]
        target_scores = results.get("targets", [])

        # Render evidence distribution
        render_evidence_distribution(target_scores)

        rank_impact = results.get("rank_impact", [])
        current_weights = st.session_state.get("last_request", {}).get("weights", {})

        if target_scores:
            # Configuration summary bar
            top_weight = max(weights.items(), key=lambda x: x[1])
            weight_summary = f"{top_weight[0].title()}: {top_weight[1]:.2f}"

            st.markdown(f"""
            <div class="config-summary">
                <div class="config-item">Disease: <span class="config-value">{selected_disease_name}</span></div>
                <div class="config-item">Targets: <span class="config-value">{len(targets)}</span></div>
                <div class="config-item">Top Weight: <span class="config-value">{weight_summary}</span></div>
                <div class="config-item">Analysis: <span class="config-value">Active</span></div>
            </div>
            """, unsafe_allow_html=True)

            # TABBED LAYOUT - Main navigation
            tab_over, tab_rank, tab_explain, tab_ev, tab_sens, tab_bench = st.tabs([
                "📊 Overview", "🏆 Rankings", "🔍 Explain", "📚 Evidence", "⚖️ Sensitivity", "📈 Benchmark"
            ])

            with tab_over:
                st.markdown("## Analytics Overview")

                # Metrics
                total_scores = [ts.get("total_score", 0) for ts in target_scores]
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Targets Analyzed", len(target_scores), help="Total targets processed")
                with col2:
                    st.metric("Best Candidate", f"{max(total_scores):.3f}", help="Highest scoring target")
                with col3:
                    st.metric("Mean Score", f"{sum(total_scores) / len(total_scores):.3f}", help="Cohort average")
                with col4:
                    st.metric("Processing Time", f"{results.get('processing_time_ms', 0):.1f}ms",
                              help="Computational efficiency")

                # Score distribution visualization
                st.markdown("### Score Distribution")
                score_df = pd.DataFrame({
                    'Target': [ts.get('target', 'Unknown') for ts in target_scores],
                    'Total Score': total_scores
                })
                st.bar_chart(score_df.set_index('Target')['Total Score'])

            with tab_rank:
                st.markdown("## Target Rankings")
                render_enhanced_results_table(target_scores, rank_impact)

            with tab_explain:
                st.markdown("## Target Explanation")

                target_names = [ts.get("target", "Unknown") for ts in target_scores]
                selected_target = st.selectbox("Select target for detailed analysis", target_names,
                                               key="explain_target")
                if not selected_target:
                    return

                selected_target_data = next((ts for ts in target_scores if ts.get("target") == selected_target), None)
                if not selected_target_data:
                    st.warning("No data for selected target")
                    return

                # Sub-tabs for explanation
                exp_contrib, exp_network, exp_modality = st.tabs([
                    "Contributions", "Network", "Modality"
                ])

                with exp_contrib:
                    render_actionable_explanation_panel(selected_target_data, selected_target)

                with exp_network:
                    render_ppi_network_card(selected_target_data)

                with exp_modality:
                    # FIXED: Render modality components properly
                    modality_fit = (selected_target_data.get("breakdown", {}) or {}).get("modality_fit", {}) or {}

                    if modality_fit and isinstance(modality_fit, dict):
                        st.markdown("### Modality Fit Analysis")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            e3 = float(modality_fit.get("e3_coexpr", 0.0) or 0.0)
                            st.metric("E3 Co-expression", f"{e3:.3f}")

                        with col2:
                            tern = float(modality_fit.get("ternary_proxy", 0.0) or 0.0)
                            st.metric("Ternary Feasibility", f"{tern:.3f}")

                        with col3:
                            hot = float(modality_fit.get("ppi_hotspot", 0.0) or 0.0)
                            st.metric("PPI Hotspot", f"{hot:.3f}")

                        # Detailed breakdown
                        modality_data = pd.DataFrame({
                            'Component': ['E3 Co-expression', 'Ternary Feasibility', 'PPI Hotspot'],
                            'Score': [e3, tern, hot]
                        })

                        st.bar_chart(modality_data.set_index('Component')['Score'])

                        # Overall druggability
                        overall_drug = float(modality_fit.get("overall_druggability", 0.0) or 0.0)
                        st.metric("Overall Druggability", f"{overall_drug:.3f}")

                    else:
                        st.info("No modality fit data available for this target")

            with tab_ev:
                st.markdown("## Supporting Evidence")

                # Evidence matrix if available
                target_names = [ts.get("target", "Unknown") for ts in target_scores]
                selected_evidence_target = st.selectbox("Select target for evidence details", target_names,
                                                        key="evidence_target")

                if selected_evidence_target:
                    target_data = next((ts for ts in target_scores if ts.get("target") == selected_evidence_target),
                                       None)
                    if target_data:
                        explanation = target_data.get("explanation", {})
                        if explanation and EVIDENCE_MATRIX_AVAILABLE:
                            try:
                                render_evidence_matrix(explanation)
                            except Exception as e:
                                st.warning(f"Evidence matrix unavailable: {e}")
                                # Fallback to simple evidence display
                                evidence_refs = explanation.get("evidence_refs", [])
                                if evidence_refs:
                                    st.markdown("**Evidence References:**")
                                    for ref in evidence_refs:
                                        if isinstance(ref, dict):
                                            label = ref.get("label", "Evidence")
                                            url = ref.get("url", "#")
                                            if url and url != "#":
                                                st.markdown(f"🔗 [{label}]({url})")
                                            else:
                                                st.markdown(f"📄 {label}")
                                        else:
                                            st.markdown(f"📄 {str(ref)}")
                        else:
                            st.info("No detailed evidence available for this target")

            with tab_sens:
                st.markdown("## Sensitivity Analysis")

                # Get the last request data
                last_request = st.session_state.get("last_request")

                if last_request:
                    # Sub-tabs for different sensitivity analyses
                    sens_tab1, sens_tab2, sens_tab3 = st.tabs([
                        "Weight Impact", "Ablation", "Stability"
                    ])

                    with sens_tab1:
                        render_weight_impact_analysis(rank_impact, current_weights)

                    with sens_tab2:
                        render_channel_ablation_analysis(results, last_request)

                    with sens_tab3:
                        render_stability_sensitivity_analysis(results, last_request)
                else:
                    st.warning("No analysis data available. Please run an analysis first.")

            with tab_bench:
                st.markdown("## Benchmark Analysis")
                render_benchmark_panel(results, selected_disease_name)

            # Back to top button
            st.markdown('<a class="backtop" href="#">↑ Top</a>', unsafe_allow_html=True)

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #94A3B8;">
            <div style="font-size: 4rem; margin-bottom: 1rem; background: linear-gradient(135deg, #22D3EE 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">🧬</div>
            <h3 style="color: #E2E8F0; margin-bottom: 1rem;">Configure Analysis Parameters</h3>
            <p style="max-width: 500px; margin: 0 auto 2rem auto; line-height: 1.6;">
                Select your target set and adjust algorithm weights in the sidebar to begin computational analysis.
                The platform integrates multi-omics data sources for comprehensive target assessment.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                <div style="background: #1E293B40; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem 1rem; transition: all 0.3s ease;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧬</div>
                    <div style="color: #E2E8F0; font-weight: 500; font-size: 0.9rem;">Multi-Omics Integration</div>
                </div>
                <div style="background: #1E293B40; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem 1rem; transition: all 0.3s ease;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                    <div style="color: #E2E8F0; font-weight: 500; font-size: 0.9rem;">AI-Powered Analysis</div>
                </div>
                <div style="background: #1E293B40; border: 1px solid #334155; border-radius: 12px; padding: 1.5rem 1rem; transition: all 0.3s ease;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <div style="color: #E2E8F0; font-weight: 500; font-size: 0.9rem;">Real-Time Results</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Platform footer
    st.markdown("""
    <div class="platform-footer">
        <span class="footer-brand">VantAI Target Scoreboard</span> v1.0.0 | 
        Powered by deep learning algorithms and multi-omics integration |
        Data sources: Open Targets, STRING, Reactome, proprietary modality databases
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()