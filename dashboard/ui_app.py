# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.



# Import enhanced components with error handling
import streamlit as st
st.set_page_config(
    page_title="VantAI Target Scoreboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",)

import sys
from pathlib import Path
import os
import requests
import pandas as pd
import io
sys.path.append(str(Path(__file__).parent.parent))

# these try/imports come AFTER set_page_config
try:
    from app.data_access.chembl import chembl_client
    from app.data_access.patent_landscape import patent_analyzer
    CHEMBL_AVAILABLE = True
except ImportError:
    print("ChEMBL integration not available")
    CHEMBL_AVAILABLE = False

try:
    from dashboard.components.network_viz import InteractiveNetworkViz
    NETWORK_VIZ_AVAILABLE = True
except ImportError:
    print("Network visualization not available")
    NETWORK_VIZ_AVAILABLE = False

ADVANCED_FEATURES_AVAILABLE = CHEMBL_AVAILABLE and NETWORK_VIZ_AVAILABLE

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

# API configuration
API_PORT = os.getenv('API_PORT', '8001')
API_BASE_URL = f"http://localhost:{API_PORT}"

# Professional VantAI CSS
def load_professional_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

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

    /* Sidebar Professional Styling */
    .css-1d391kg {{
        background: {VANTAI_THEME['bg_surface']};
        border-right: 1px solid {VANTAI_THEME['bg_border']};
        backdrop-filter: blur(10px);
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {VANTAI_THEME['text_primary']} !important;
        font-weight: 600;
        letter-spacing: -0.025em;
    }}

    /* Main Header */
    .platform-header {{
        background: {VANTAI_THEME['gradient_surface']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: {VANTAI_THEME['shadow_card']};
        position: relative;
        overflow: hidden;
    }}

    .platform-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: {VANTAI_THEME['gradient_primary']};
    }}

    .platform-title {{
        font-size: 2.75rem;
        font-weight: 700;
        color: {VANTAI_THEME['text_primary']};
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }}

    .platform-subtitle {{
        color: {VANTAI_THEME['text_secondary']};
        font-size: 1.125rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    /* Section Headers */
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

    /* Metric Cards */
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

    /* Enhanced Dataframe Styling */
    .stDataFrame {{
        background: transparent !important;
    }}

    .stDataFrame > div {{
        background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%) !important;
        border: 1px solid {VANTAI_THEME['bg_border']} !important;
        border-radius: 8px !important;
    }}

    .stDataFrame [data-testid="stDataFrameResizeHandle"] {{
        display: none !important;
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

    /* Sliders */
    .stSlider > div > div > div > div {{
        background: {VANTAI_THEME['accent_cyan']};
    }}

    .stSlider > div > div > div {{
        background: {VANTAI_THEME['bg_border']};
    }}

    /* Sidebar Sections */
    .sidebar-section {{
        margin: 1.5rem 0;
        padding: 1rem 0;
        border-bottom: 1px solid {VANTAI_THEME['bg_border']};
    }}

    .sidebar-section:last-child {{
        border-bottom: none;
    }}

    .sidebar-title {{
        color: {VANTAI_THEME['text_primary']};
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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

    .component-score {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid {VANTAI_THEME['bg_border']};
    }}

    .component-score:last-child {{
        border-bottom: none;
    }}

    .component-name {{
        color: {VANTAI_THEME['text_secondary']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }}

    .component-value {{
        font-weight: 700;
        font-size: 1.1rem;
        font-variant-numeric: tabular-nums;
    }}

    /* Score Classes */
    .score-high {{
        color: {VANTAI_THEME['success']};
    }}

    .score-medium {{
        color: {VANTAI_THEME['warning']};
    }}

    .score-low {{
        color: {VANTAI_THEME['danger']};
    }}

    /* Status Messages */
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

    /* Loading States */
    .stSpinner > div {{
        border-top-color: {VANTAI_THEME['accent_cyan']} !important;
    }}

    /* Progress Indicators */
    .stProgress > div > div > div > div {{
        background: {VANTAI_THEME['gradient_primary']};
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .platform-title {{
            font-size: 2rem;
        }}

        .metrics-grid {{
            grid-template-columns: 1fr;
        }}

        .data-table th,
        .data-table td {{
            padding: 0.75rem 1rem;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)
# API configuration
API_PORT = os.getenv('API_PORT', '8001')
API_BASE_URL = f"http://localhost:{API_PORT}"


def call_api(endpoint, method="GET", data=None):
    """Call the FastAPI backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Ensure the backend service is running.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None


def handle_export_download(export_format, request_data):
    """Handle export download via API call."""
    try:
        export_url = f"{API_BASE_URL}/export/{export_format}"
        response = requests.post(export_url, json=request_data, timeout=30)

        if response.status_code == 200:
            # Get filename from Content-Disposition header
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = f'vantai_export.{export_format}'
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')

            return response.content, filename
        else:
            st.error(f"Export failed: {response.status_code}")
            return None, None

    except Exception as e:
        st.error(f"Export error: {e}")
        return None, None


def render_why_panel(target_data, selected_target):
    """
    Render 'Why ↑/↓?' explanation panel using proper Streamlit components.
    """
    breakdown = target_data.get("breakdown", {})
    evidence_refs = target_data.get("evidence_refs", [])

    # Use Streamlit container for better layout
    with st.container():
        st.markdown(f"### Why Ranked Here? • {selected_target}")
        st.caption("Click evidence links for detailed sources (PMID/database references)")

        # Channel contributions analysis
        contributions = []

        # Genetics contribution
        genetics_score = breakdown.get("genetics", 0)
        if genetics_score > 0.5:
            impact = "High"
            impact_emoji = "↑"
            reason = "Strong genetic association with disease"
        elif genetics_score > 0.3:
            impact = "Medium"
            impact_emoji = "→"
            reason = "Moderate genetic support"
        else:
            impact = "Low"
            impact_emoji = "↓"
            reason = "Limited genetic evidence"

        genetics_evidence = [ref for ref in evidence_refs if "genetics" in ref.lower()][:3]
        contributions.append({
            "channel": "Genetics",
            "impact": f"{impact_emoji} {impact}",
            "score": genetics_score,
            "reason": reason,
            "evidence": genetics_evidence
        })

        # PPI contribution
        ppi_score = breakdown.get("ppi_proximity", 0)
        if ppi_score > 0.6:
            impact = "High"
            impact_emoji = "↑"
            reason = "Central hub in disease network"
        else:
            impact = "Peripheral"
            impact_emoji = "↓"
            reason = "Peripheral network position"

        ppi_evidence = [ref for ref in evidence_refs if "string" in ref.lower() or "ppi" in ref.lower()][:2]
        contributions.append({
            "channel": "Network",
            "impact": f"{impact_emoji} {impact}",
            "score": ppi_score,
            "reason": reason,
            "evidence": ppi_evidence
        })

        # Modality contribution
        modality_fit = breakdown.get("modality_fit", {})
        overall_mod = modality_fit.get("overall_druggability", 0) if modality_fit else 0
        if overall_mod > 0.5:
            impact = "Druggable"
            impact_emoji = "↑"
            reason = "Good modality fit for targeting"
        else:
            impact = "Challenging"
            impact_emoji = "↓"
            reason = "Limited druggability options"

        modality_evidence = [ref for ref in evidence_refs if "vantai" in ref.lower()][:2]
        contributions.append({
            "channel": "Modality",
            "impact": f"{impact_emoji} {impact}",
            "score": overall_mod,
            "reason": reason,
            "evidence": modality_evidence
        })

        # Render each contribution using Streamlit components
        for contrib in contributions:
            with st.expander(f"{contrib['channel']}: {contrib['impact']} ({contrib['score']:.3f})", expanded=True):
                st.write(contrib['reason'])

                if contrib['evidence']:
                    st.markdown("**Evidence:**")
                    badge_html = render_badges(contrib['evidence'])
                    if badge_html:
                        st.markdown(badge_html, unsafe_allow_html=True)


# 2. ADD this new function:
def render_badges(evidence_list):
    """
    Create clickable badges for PMID/DB references using Markdown.
    """
    if not evidence_list:
        return ""

    badges = []
    for evidence in evidence_list:
        # Extract PMID or database info
        if "PMID:" in evidence:
            pmid = evidence.split("PMID:")[1].split()[0]
            link_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            badge_text = f"PMID:{pmid}"
        elif "STRING:" in evidence:
            link_url = "https://string-db.org/"
            badge_text = "STRING-DB"
        elif "Reactome:" in evidence:
            link_url = "https://reactome.org/"
            badge_text = "Reactome"
        elif "VantAI:" in evidence:
            link_url = "#"  # Internal data
            badge_text = "VantAI-DB"
        else:
            link_url = "#"
            badge_text = evidence[:20] + "..." if len(evidence) > 20 else evidence

        # Create markdown badge with external link
        if link_url != "#":
            badge = f'<a href="{link_url}" target="_blank" style="display: inline-block; background: #1E293B; color: #22D3EE; padding: 0.25rem 0.5rem; border-radius: 4px; text-decoration: none; font-size: 0.8rem; font-weight: 500; margin: 0.125rem;">{badge_text}</a>'
        else:
            badge = f'<span style="display: inline-block; background: #374151; color: #9CA3AF; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 500; margin: 0.125rem;">{badge_text}</span>'

        badges.append(badge)

    return " ".join(badges)


import streamlit as st
import pandas as pd
from typing import Dict, List, Any


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

        available = score is not None
        score = 0.0 if score is None else float(score)
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
                    with st.container():
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

        # Legend
        st.markdown("**Legend:** 📈 Moved up • 📉 Moved down • ➡️ No change")

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
            rank_indicator = f"📈 {i} (+{delta})"
        elif movement == "down":
            rank_indicator = f"📉 {i} (-{abs(delta)})"
        else:
            rank_indicator = f"➡️ {i}"

        table_data.append({
            "Rank": rank_indicator,
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
        "Rank": st.column_config.TextColumn("Rank", width="small", help="Current rank with movement vs default weights"),
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


def integrate_fixed_components_in_main():
    """
    Integration guide for main dashboard function.
    Replace the existing render_why_panel and HTML-based components with these fixed versions.
    """

    integration_instructions = """
    # In your main dashboard function, replace these calls:

    # OLD (broken HTML rendering):
    # render_why_panel(target_data, selected_target)

    # NEW (fixed Streamlit components):
    if target_data:
        render_actionable_explanation_panel(target_data, selected_target)

    # Also replace ranking impact analysis:
    # OLD: Any raw HTML for ranking changes

    # NEW: 
    if "scoring_results" in st.session_state:
        results = st.session_state["scoring_results"]
        rank_impact = results.get("rank_impact", [])
        current_weights = st.session_state.get("last_request", {}).get("weights", {})
        if rank_impact:
            render_ranking_impact_analysis(rank_impact, current_weights)

    # Replace results table:
    # NEW:
    render_enhanced_results_table(target_scores, rank_impact)
    """

    return integration_instructions

def render_actionable_explanation_panel(target_data: Dict, selected_target: str):
    """
    Render actionable explanation panel with clickable contributions and evidence.
    Fixes the raw HTML rendering issue by using proper Streamlit components.
    """
    if not target_data:
        st.info("No explanation data available for this target")
        return

    # Extract explanation data
    explanation = target_data.get("explanation", {}) or {}
    is_error_state = str(target_data.get("data_version", "")).lower().startswith("error")
    no_contribs = not explanation.get("contributions")
    if is_error_state or no_contribs:
        breakdown = target_data.get("breakdown", {}) or {}
        explanation = _build_fallback_explanation(
            selected_target,
                       breakdown,
                        target_data.get("evidence_refs", []) or []
            )
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
                            # External clickable link
                            st.markdown(
                                f'<a href="{url}" target="_blank" style="display: inline-block; background: #1E293B; color: #22D3EE; padding: 0.5rem 0.75rem; border-radius: 6px; text-decoration: none; font-size: 0.8rem; font-weight: 500; margin: 0.25rem 0; border: 1px solid #22D3EE40; width: 100%; text-align: center;">{label}</a>',
                                unsafe_allow_html=True)
                        else:
                            # Internal/unavailable - use button style
                            st.button(label, disabled=True, key=f"evidence_{i}_{ref_type}")
        else:
            raw_refs = target_data.get("evidence_refs", [])
            if raw_refs:
                st.markdown(render_badges(raw_refs), unsafe_allow_html=True)
            else:
                st.info("No evidence references available")

        # Summary metrics
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


def render_delta_ranking_view(rank_impact: List[Dict], current_weights: Dict[str, float]):
    """
    Render delta ranking view showing ranking changes.

    Args:
        rank_impact: List of ranking impact objects from API
        current_weights: Current weight configuration
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
        return

    with st.container():
        st.markdown("### Ranking Impact Analysis")
        st.caption("How current weights change rankings vs. default configuration")

        # Create grid layout for target cards
        cols = st.columns(3)

        for i, impact in enumerate(rank_impact[:9]):  # Show top 9 targets
            with cols[i % 3]:
                target = impact["target"]
                rank_baseline = impact["rank_baseline"]
                rank_current = impact["rank_current"]
                delta = impact["delta"]
                movement = impact["movement"]

                # Movement indicators and colors
                if movement == "up":
                    indicator = "↗"
                    color = "#34D399"  # Green
                    message = f"Rank {rank_baseline} → {rank_current} (+{delta})"
                elif movement == "down":
                    indicator = "↘"
                    color = "#F87171"  # Red
                    message = f"Rank {rank_baseline} → {rank_current} (-{abs(delta)})"
                else:
                    indicator = "→"
                    color = "#94A3B8"  # Gray
                    message = f"Rank {rank_current} (unchanged)"

                # Create card for each target
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%);
                        border: 1px solid #1E293B;
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        text-align: center;
                    ">
                        <div style="
                            font-size: 1.5rem;
                            margin-bottom: 0.5rem;
                        ">{indicator}</div>
                        <div style="
                            font-weight: 600;
                            color: #E2E8F0;
                            margin-bottom: 0.5rem;
                        ">{target}</div>
                        <div style="
                            color: {color};
                            font-size: 0.85rem;
                            font-weight: 500;
                        ">{message}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Summary information
        if len(rank_impact) > 9:
            st.info(f"Showing top 9 targets. Total targets analyzed: {len(rank_impact)}")

        # Legend
        st.markdown("""
        **Legend:** ↗ Improved ranking • ↘ Decreased ranking • → No change
        """)


def render_weight_impact_summary(current_weights: Dict[str, float]):
    """Render summary of weight changes."""
    default_weights = {
        "genetics": 0.35,
        "ppi": 0.25,
        "pathway": 0.20,
        "safety": 0.10,
        "modality_fit": 0.10
    }

    # Find significant changes
    changes = []
    for channel, default_val in default_weights.items():
        current_val = current_weights.get(channel, default_val)
        diff = current_val - default_val
        if abs(diff) > 0.05:  # Significant change
            direction = "increased" if diff > 0 else "decreased"
            changes.append(f"{channel.replace('_', ' ').title()} {direction} by {abs(diff):.2f}")

    if changes:
        st.info(f"Weight changes: {', '.join(changes[:2])}" +
                (f" (+{len(changes) - 2} more)" if len(changes) > 2 else ""))


# Integration functions for main dashboard
def integrate_explanation_components():
    """
    Integration instructions for main dashboard.

    Add these calls in the appropriate sections of dashboard/app.py:
    """

    # 1. In target details section, replace existing "Why ranked here?" with:
    explanation_integration = '''
    # Add explanation panel
    if target_data and target_data.get("explanation"):
        render_actionable_explanation_panel(
            target_data["explanation"], 
            selected_target
        )
    '''

    # 2. After analytics overview, add delta ranking:
    delta_ranking_integration = '''
    # Add delta ranking analysis
    if "scoring_results" in st.session_state:
        results = st.session_state["scoring_results"]
        rank_impact = results.get("rank_impact", [])
        if rank_impact:
            render_delta_ranking_view(rank_impact, current_weights)
    '''

    # 3. In sidebar, add weight impact summary:
    weight_summary_integration = '''
    # Add weight impact summary
    render_weight_impact_summary(weights)
    '''

    return {
        "explanation": explanation_integration,
        "delta_ranking": delta_ranking_integration,
        "weight_summary": weight_summary_integration
    }


# Enhanced results table with ranking information
def render_enhanced_results_table_with_ranking(target_scores, rank_impact=None):
    """Enhanced results table with ranking change indicators."""
    if not target_scores:
        return

    sorted_targets = sorted(target_scores, key=lambda x: x.get("total_score", 0), reverse=True)

    # Create ranking lookup
    rank_lookup = {}
    if rank_impact:
        rank_lookup = {item["target"]: item for item in rank_impact}

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
            rank_indicator = f"↗ +{delta}"
        elif movement == "down":
            rank_indicator = f"↘ -{abs(delta)}"
        else:
            rank_indicator = "→"

        rank_display = f"{i} {rank_indicator}" if rank_info else str(i)

        table_data.append({
            "Rank": rank_display,
            "Target": target,
            "Total Score": ts.get("total_score", 0),
            "Genetics": breakdown.get("genetics", 0),
            "PPI (RWR)": breakdown.get("ppi_proximity", 0),
            "Pathway": breakdown.get("pathway_enrichment", 0),
            "Modality": modality_fit.get("overall_druggability", 0) if modality_fit else 0
        })

    df = pd.DataFrame(table_data)

    column_config = {
        "Rank": st.column_config.TextColumn("Rank", width="small",
                                            help="Current rank with movement indicator"),
        "Target": st.column_config.TextColumn("Target", width="medium"),
        "Total Score": st.column_config.NumberColumn("Total Score", format="%.3f", width="medium"),
        "Genetics": st.column_config.NumberColumn("Genetics", format="%.3f", width="medium"),
        "PPI (RWR)": st.column_config.NumberColumn("PPI (RWR)", format="%.3f", width="medium",
                                                   help="Network proximity with Random Walk Restart"),
        "Pathway": st.column_config.NumberColumn("Pathway", format="%.3f", width="medium"),
        "Modality": st.column_config.NumberColumn("Modality", format="%.3f", width="medium")
    }

    st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(400, (len(df) + 1) * 35 + 40)
    )


def render_data_version_footer(results, processing_time_ms):
    """Render data version and cache status footer."""
    target_scores = results.get("targets", [])
    if not target_scores:
        return ""

    # Get data version from first target
    data_version = (
            results.get("data_version") or
            (target_scores[0].get("data_version") if target_scores else "Unknown")
    )
    # Parse version components
    version_parts = data_version.split(" | ")
    version_display = " • ".join(version_parts)

    # Determine cache status
    if processing_time_ms < 50:
        cache_status = "cached"
        cache_color = VANTAI_THEME['success']
        cache_icon = "🟢"
    elif processing_time_ms < 200:
        cache_status = "partial cache"
        cache_color = VANTAI_THEME['warning']
        cache_icon = "🟡"
    else:
        cache_status = "fresh fetch"
        cache_color = VANTAI_THEME['accent_cyan']
        cache_icon = "🔄"

    footer_html = f"""
    <div style="
        background: {VANTAI_THEME['bg_surface']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        margin: 2rem 0 1rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        color: {VANTAI_THEME['text_muted']};
        flex-wrap: wrap;
        gap: 1rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: {VANTAI_THEME['text_secondary']}; font-weight: 500;">Data Sources:</span>
            <span style="font-family: monospace; color: {VANTAI_THEME['text_primary']};">
                {version_display}
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.25rem;">
                <span>{cache_icon}</span>
                <span style="color: {cache_color}; font-weight: 500;">{cache_status}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.25rem;">
                <span>⚡</span>
                <span style="color: {VANTAI_THEME['text_secondary']};">
                    {processing_time_ms:.1f} ms
                </span>
            </div>
        </div>
    </div>
    """
    return footer_html


def render_version_tooltip():
    """Render tooltip explaining data source versions."""
    return f"""
    <div style="
        background: {VANTAI_THEME['bg_card']};
        border: 1px solid {VANTAI_THEME['bg_border']};
        border-radius: 6px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: {VANTAI_THEME['text_secondary']};
        max-width: 400px;
    ">
        <div style="font-weight: 600; color: {VANTAI_THEME['text_primary']}; margin-bottom: 0.5rem;">
            Data Source Information
        </div>
        <div style="margin-bottom: 0.25rem;">
            • <strong>OT:</strong> Open Targets Platform (genetics, safety)
        </div>
        <div style="margin-bottom: 0.25rem;">
            • <strong>STRING:</strong> Protein-protein interaction network
        </div>
        <div style="margin-bottom: 0.25rem;">
            • <strong>Reactome:</strong> Biological pathway database
        </div>
        <div>
            • <strong>VantAI:</strong> Proprietary modality scoring
        </div>
    </div>
    """


def compute_delta_rankings(target_scores, current_weights, default_weights):
    """
    Compute ranking changes when switching from current to default weights.

    Returns:
        List of dicts with delta analysis per target
    """

    # Current ranking
    current_ranking = {
        target['target']: i + 1
        for i, target in enumerate(
            sorted(target_scores, key=lambda x: x['total_score'], reverse=True)
        )
    }

    # Recompute scores with default weights
    default_target_scores = []
    for target_data in target_scores:
        breakdown = target_data.get('breakdown', {})

        # Extract channel scores
        genetics = breakdown.get('genetics', 0)
        ppi = breakdown.get('ppi_proximity', 0)
        pathway = breakdown.get('pathway_enrichment', 0)
        safety = breakdown.get('safety_off_tissue', 0)
        modality_fit = breakdown.get('modality_fit', {})
        modality_overall = modality_fit.get('overall_druggability', 0) if modality_fit else 0

        # Compute default weighted score
        default_score = (
                genetics * default_weights['genetics'] +
                ppi * default_weights['ppi'] +
                pathway * default_weights['pathway'] +
                (1 - safety) * default_weights['safety'] +  # Safety inverted
                modality_overall * default_weights['modality_fit']
        )

        default_target_scores.append({
            'target': target_data['target'],
            'total_score': default_score,
            'original_score': target_data['total_score']
        })

    # Default ranking
    default_ranking = {
        target['target']: i + 1
        for i, target in enumerate(
            sorted(default_target_scores, key=lambda x: x['total_score'], reverse=True)
        )
    }

    # Compute deltas
    delta_analysis = []
    for target_data in target_scores:
        target = target_data['target']
        current_rank = current_ranking[target]
        default_rank = default_ranking[target]
        delta_rank = current_rank - default_rank  # Negative = moved up, positive = moved down

        delta_analysis.append({
            'target': target,
            'current_rank': current_rank,
            'default_rank': default_rank,
            'delta_rank': delta_rank,
            'current_score': target_data['total_score'],
            'default_score': next(t['total_score'] for t in default_target_scores if t['target'] == target)
        })

    return sorted(delta_analysis, key=lambda x: x['current_rank'])


def render_delta_rank_view(target_scores, current_weights):
    """
    Render delta ranking comparison using proper Streamlit components.
    """
    default_weights = {
        "genetics": 0.35,
        "ppi": 0.25,
        "pathway": 0.20,
        "safety": 0.10,
        "modality_fit": 0.10
    }

    # Check if weights are different from default
    weights_changed = any(
        abs(current_weights.get(k, 0) - default_weights[k]) > 0.01
        for k in default_weights
    )

    if not weights_changed:
        return

    # Compute delta analysis (keep your existing logic here)
    delta_analysis = compute_delta_rankings(target_scores, current_weights, default_weights)

    # Render using Streamlit components instead of HTML
    with st.container():
        st.markdown("### Ranking Impact Analysis")
        st.caption("Current vs Default Weights")

        # Create columns for grid layout
        cols = st.columns(3)

        for i, delta in enumerate(delta_analysis[:6]):  # Show top 6 targets
            with cols[i % 3]:
                target = delta['target']
                delta_rank = delta['delta_rank']

                if delta_rank < 0:  # Moved up
                    arrow = "↗"
                    delta_text = f"Rank {delta['default_rank']} → {delta['current_rank']} (+{abs(delta_rank)})"
                    color = "🟢"
                elif delta_rank > 0:  # Moved down
                    arrow = "↘"
                    delta_text = f"Rank {delta['default_rank']} → {delta['current_rank']} (-{abs(delta_rank)})"
                    color = "🔴"
                else:  # No change
                    arrow = "→"
                    delta_text = f"Rank {delta['current_rank']} (unchanged)"
                    color = "⚪"

                score_change = delta['current_score'] - delta['default_score']
                score_change_text = f"{score_change:+.3f}" if abs(score_change) > 0.001 else "±0.000"

                with st.container():
                    st.markdown(f"**{target}** {arrow}")
                    st.write(f"{color} {delta_text}")
                    st.caption(f"Score: {score_change_text}")

        st.info("**Legend:** ↗ Improved ranking • ↘ Decreased ranking • → No change")


def render_metric_card(label, value, description=None):
    """Render a professional metric card."""
    description_html = f"<div class='metric-description'>{description}</div>" if description else ""

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {description_html}
    </div>
    """


def get_score_class(score):
    """Get CSS class for score styling."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    else:
        return "score-low"


def render_results_table(target_scores):
    """Render professional results table using native Streamlit components."""
    if not target_scores:
        return

    sorted_targets = sorted(target_scores, key=lambda x: x.get("total_score", 0), reverse=True)

    # Create DataFrame for Streamlit table
    table_data = []
    for i, ts in enumerate(sorted_targets, 1):
        breakdown = ts.get("breakdown", {})
        modality_fit = breakdown.get("modality_fit", {})

        total_score = ts.get("total_score", 0)
        genetics_score = breakdown.get("genetics", 0)
        ppi_score = breakdown.get("ppi_proximity", 0)
        pathway_score = breakdown.get("pathway_enrichment", 0)
        modality_score = modality_fit.get("overall_druggability", 0) if modality_fit else 0

        # Add rank based on position
        rank_display = "1st" if i == 1 else "2nd" if i == 2 else "3rd" if i == 3 else f"{i}th"

        table_data.append({
            "Rank": rank_display,
            "Target": ts.get('target', 'Unknown'),
            "Total Score": total_score,
            "Genetics": genetics_score,
            "PPI Proximity": ppi_score,
            "Pathway": pathway_score,
            "Modality Fit": modality_score
        })

    # Use Streamlit's dataframe with custom styling
    df = pd.DataFrame(table_data)

    # Configure column display
    column_config = {
        "Rank": st.column_config.TextColumn("Rank", width="small"),
        "Target": st.column_config.TextColumn("Target", width="medium"),
        "Total Score": st.column_config.NumberColumn("Total Score", format="%.3f", width="medium"),
        "Genetics": st.column_config.NumberColumn("Genetics", format="%.3f", width="medium"),
        "PPI Proximity": st.column_config.NumberColumn("PPI Proximity", format="%.3f", width="medium"),
        "Pathway": st.column_config.NumberColumn("Pathway", format="%.3f", width="medium"),
        "Modality Fit": st.column_config.NumberColumn("Modality Fit", format="%.3f", width="medium")
    }

    # Display the table with custom configuration
    st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(400, (len(df) + 1) * 35 + 40)  # Dynamic height
    )


# Main application
def main():
    """Main dashboard function with fixed components integrated."""

    # Load professional theme
    load_professional_css()

    # Platform header
    st.markdown("""
    <div class="platform-header">
        <div class="platform-title">VantAI Target Scoreboard</div>
        <div class="platform-subtitle">
            Advanced computational platform for modality-aware target prioritization using multi-omics integration
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)

        # Disease selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Disease Context**")
        disease_options = {
            "Non-small cell lung carcinoma": "EFO_0000305",
            "Breast carcinoma": "EFO_0000305",
            "Colorectal carcinoma": "EFO_0000305"
        }

        selected_disease_name = st.selectbox(
            "Select Disease",
            list(disease_options.keys()),
            label_visibility="collapsed"
        )
        disease_id = disease_options[selected_disease_name]
        st.markdown('</div>', unsafe_allow_html=True)

        # Target input
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Target Selection**")

        target_sets = {
            "NSCLC Targets": ["EGFR", "ERBB2", "MET", "ALK", "KRAS"],
            "Oncogenes": ["EGFR", "ERBB2", "MET", "ALK", "BRAF", "PIK3CA"],
            "Tumor Suppressors": ["TP53", "RB1", "PTEN"],
            "Custom": []
        }

        selected_set = st.selectbox("Target Set", list(target_sets.keys()))

        if selected_set == "Custom":
            targets_input = st.text_area(
                "Enter targets (one per line)",
                value="EGFR\nERBB2\nMET\nALK\nKRAS",
                height=100
            )
            targets = [t.strip().upper() for t in targets_input.split("\n") if t.strip()]
        else:
            targets = target_sets[selected_set]
            st.markdown(f"*Targets:* {', '.join(targets)}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Scoring weights
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**Algorithm Weights**")

        genetics_weight = st.slider("Genetics", 0.0, 1.0, 0.35, 0.05)
        ppi_weight = st.slider("PPI Proximity", 0.0, 1.0, 0.25, 0.05)
        pathway_weight = st.slider("Pathway", 0.0, 1.0, 0.20, 0.05)
        safety_weight = st.slider("Safety", 0.0, 1.0, 0.10, 0.05)
        modality_weight = st.slider("Modality Fit", 0.0, 1.0, 0.10, 0.05)

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
        st.markdown('</div>', unsafe_allow_html=True)

        # Execute analysis
        if st.button("Execute Analysis", type="primary"):
            if not targets:
                st.error("Please select or enter targets for analysis")
                return

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

        # Export functionality - keep your existing export section
        if "scoring_results" in st.session_state and "last_request" in st.session_state:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**Export Results**")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("JSON", help="Download full results"):
                    with st.spinner("Generating JSON..."):
                        content, filename = handle_export_download('json', st.session_state["last_request"])
                        if content:
                            st.download_button(
                                label="Download JSON",
                                data=content,
                                file_name=filename,
                                mime="application/json",
                                key="json_download"
                            )

            with col2:
                if st.button("CSV", help="Download CSV"):
                    with st.spinner("Generating CSV..."):
                        content, filename = handle_export_download('csv', st.session_state["last_request"])
                        if content:
                            st.download_button(
                                label="Download CSV",
                                data=content,
                                file_name=filename,
                                mime="text/csv",
                                key="csv_download"
                            )

            st.markdown('</div>', unsafe_allow_html=True)

    # MAIN CONTENT AREA - THIS IS WHERE THE FIXES ARE INTEGRATED
    if "scoring_results" in st.session_state:
        results = st.session_state["scoring_results"]
        target_scores = results.get("targets", [])

        if target_scores:
            # Analytics overview
            st.markdown('<div class="section-header">Analytics Overview</div>', unsafe_allow_html=True)

            total_scores = [ts.get("total_score", 0) for ts in target_scores]

            metrics_html = f"""
            <div class="metrics-grid">
                {render_metric_card("Targets Analyzed", len(target_scores), "Total targets processed")}
                {render_metric_card("Best Candidate", f"{max(total_scores):.3f}", "Highest scoring target")}
                {render_metric_card("Mean Score", f"{sum(total_scores) / len(total_scores):.3f}", "Cohort average")}
                {render_metric_card("Processing Time", f"{results.get('processing_time_ms', 0):.1f}ms", "Computational efficiency")}
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)

            # FIXED: Ranking impact analysis using proper Streamlit components
            if "last_request" in st.session_state:
                current_weights = st.session_state["last_request"]["weights"]
                rank_impact = results.get("rank_impact", [])
                if rank_impact:
                    render_ranking_impact_analysis(rank_impact, current_weights)

            # FIXED: Enhanced results table with ranking indicators
            st.markdown('<div class="section-header">Target Rankings</div>', unsafe_allow_html=True)
            rank_impact = results.get("rank_impact", [])
            render_enhanced_results_table(target_scores, rank_impact)

            # FIXED: Target details with actionable explanation panel
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)

            target_names = [ts.get("target", "Unknown") for ts in target_scores]
            selected_target = st.selectbox("Select target for detailed analysis", target_names,
                                           key="target_select_detailed")

            if selected_target:
                target_data = next((ts for ts in target_scores if ts.get("target") == selected_target), None)
                if target_data:
                    # FIXED: Use the new actionable explanation panel
                    render_actionable_explanation_panel(target_data, selected_target)
                    # --- Modality Components panel (overall/protac/small_molecule’dan sonra) ---
                    modality_fit = (target_data.get("breakdown", {}) or {}).get("modality_fit", {}) or {}

                    if modality_fit:
                        # (İsteğe bağlı) Eğer overall/protac/small_molecule skorlarını da göstermek istersen:
                        # overall = float(modality_fit.get("overall_druggability", 0.0) or 0.0)
                        # protac  = float(modality_fit.get("protac_degrader", 0.0) or 0.0)
                        # small   = float(modality_fit.get("small_molecule", 0.0) or 0.0)
                        # st.markdown(f"... buraya üçlü özet kartlarını koyabilirsin ...", unsafe_allow_html=True)

                        # Bileşen skorları
                        e3 = float(modality_fit.get("e3_coexpr", 0.0) or 0.0)
                        tern = float(modality_fit.get("ternary_proxy", 0.0) or 0.0)
                        hot = float(modality_fit.get("ppi_hotspot", 0.0) or 0.0)

                        st.markdown(f"""
                        <div class="target-details">
                          <h4>Modality Components</h4>
                          <div class="component-score">
                            <span class="component-name">E3 Co-expression</span>
                            <span class="component-value {get_score_class(e3)}">{e3:.3f}</span>
                          </div>
                          <div class="component-score">
                            <span class="component-name">Ternary Feasibility</span>
                            <span class="component-value {get_score_class(tern)}">{tern:.3f}</span>
                          </div>
                          <div class="component-score">
                            <span class="component-name">PPI Hotspot</span>
                            <span class="component-value {get_score_class(hot)}">{hot:.3f}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Data version footer
            footer_html = render_data_version_footer(results, results.get('processing_time_ms', 0))
            st.markdown(footer_html, unsafe_allow_html=True)

            # Version tooltip (expandable)
            with st.expander("Data Source Details"):
                st.markdown("""
                **Data Source Information:**
                - **OT:** Open Targets Platform (genetics, safety)
                - **STRING:** Protein-protein interaction network  
                - **Reactome:** Biological pathway database
                - **VantAI:** Proprietary modality scoring
                """)

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #94A3B8;">
            <h3 style="color: #E2E8F0; margin-bottom: 1rem;">Configure Analysis Parameters</h3>
            <p style="max-width: 500px; margin: 0 auto; line-height: 1.6;">
                Select your target set and adjust algorithm weights in the sidebar to begin computational analysis.
                The platform integrates multi-omics data sources for comprehensive target assessment.
            </p>
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