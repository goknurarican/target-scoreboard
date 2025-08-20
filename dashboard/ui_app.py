# Copyright (c) 2025 Göknur Arıcan
# All rights reserved. Licensed for internal evaluation only.
# See LICENSE-EVALUATION.md for terms.

#dashboard/ui_app.py

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
from dashboard.components.explanation_panel import render_evidence_matrix

sys.path.append(str(Path(__file__).parent.parent))
try:
    from dashboard.components.explanation_panel import render_evidence_matrix

    EVIDENCE_MATRIX_AVAILABLE = True
except ImportError:
    EVIDENCE_MATRIX_AVAILABLE = False


    # Fallback implementation if component file doesn't exist
    def render_evidence_matrix(explanation):
        """Fallback evidence matrix implementation."""
        if not explanation or "evidence_refs" not in explanation:
            st.info("No evidence references available")
            return

        evidence_refs = explanation.get("evidence_refs", [])
        if not evidence_refs:
            st.info("No evidence references found")
            return

        # Simple fallback - group by type and show in tabs
        evidence_by_type = {
            "literature": [],
            "database": [],
            "proprietary": [],
            "other": []
        }

        for ref in evidence_refs:
            if isinstance(ref, dict):
                ref_type = ref.get("type", "other")
                evidence_by_type.setdefault(ref_type, []).append(ref)
            else:
                evidence_by_type["other"].append({
                    "label": str(ref),
                    "url": "#",
                    "type": "other"
                })

        # Create tabs with counts
        tab_labels = [
            f"📚 Literature ({len(evidence_by_type.get('literature', []))})",
            f"🗄️ Databases ({len(evidence_by_type.get('database', []))})",
            f"🧪 VantAI ({len(evidence_by_type.get('proprietary', []))})",
            f"⚙️ Other ({len(evidence_by_type.get('other', []))})"
        ]

        tabs = st.tabs(tab_labels)

        # Render each tab
        tab_types = ["literature", "database", "proprietary", "other"]
        for i, tab_type in enumerate(tab_types):
            with tabs[i]:
                refs = evidence_by_type.get(tab_type, [])
                if not refs:
                    st.info(f"No {tab_type} evidence available")
                    continue

                # Search filter
                search_term = st.text_input(
                    f"Search {tab_type} evidence:",
                    key=f"{tab_type}_search_fallback",
                    placeholder="Filter by label..."
                )

                # Filter references
                filtered_refs = refs
                if search_term:
                    filtered_refs = [
                        ref for ref in refs
                        if search_term.lower() in ref.get("label", "").lower()
                    ]

                if not filtered_refs:
                    st.warning(f"No {tab_type} evidence matches '{search_term}'")
                    continue

                # Show filtered count
                if search_term:
                    st.caption(f"Showing {len(filtered_refs)} of {len(refs)} references")

                # Display badges
                for ref in filtered_refs:
                    label = ref.get("label", "Unknown")
                    url = ref.get("url", "#")

                    if url and url != "#":
                        st.markdown(f"[🔗 {label}]({url})")
                    else:
                        st.markdown(f"📄 {label}")

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



def render_stability_sensitivity_analysis(results, last_request):
    """
    Render weight sensitivity and ranking stability analysis using proper Streamlit components.
    """
    with st.container():
        st.markdown('<div class="section-header">Stability & Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.caption("Analyze how ranking stability varies under weight uncertainty")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🎯 Simulate Weight Sensitivity", help="Run Monte Carlo simulation of weight perturbations"):
                st.session_state["run_simulation"] = True

        with col2:
            # Show simulation parameters
            st.markdown("""
            **Parameters:** 200 samples, Dirichlet α=80.0  
            Higher α = less weight variation
            """)

        # Run simulation if requested
        if st.session_state.get("run_simulation", False):
            with st.spinner("Running weight perturbation simulation..."):
                try:
                    # Call simulation endpoint
                    response = call_api("/simulate/weights", method="POST", data=last_request)

                    if response:
                        st.session_state["simulation_results"] = response
                        st.session_state["run_simulation"] = False
                        st.success("Simulation completed successfully!")
                    else:
                        st.error("Simulation failed - check API connection")
                        st.session_state["run_simulation"] = False

                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    st.session_state["run_simulation"] = False

        # Display simulation results if available
        if "simulation_results" in st.session_state:
            sim_data = st.session_state["simulation_results"]

            # Extract data safely
            simulation_results = sim_data.get("simulation_results", {})
            stability_data = simulation_results.get("stability", {})
            kendall_tau = simulation_results.get("kendall_tau_mean", 0.0)
            samples_count = simulation_results.get("samples", 0)
            weight_stats = simulation_results.get("weight_stats", {})

            if stability_data:
                # Overall stability metrics
                st.markdown("#### Ranking Stability Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Kendall tau interpretation
                    if kendall_tau >= 0.9:
                        stability_level = "Very Stable"
                        stability_color = "#34D399"
                    elif kendall_tau >= 0.8:
                        stability_level = "Stable"
                        stability_color = "#22D3EE"
                    elif kendall_tau >= 0.6:
                        stability_level = "Moderate"
                        stability_color = "#F59E0B"
                    else:
                        stability_level = "Unstable"
                        stability_color = "#F87171"

                    st.metric("Rank Correlation", f"{kendall_tau:.3f}", help="Kendall's τ (higher = more stable)")
                    st.markdown(
                        f"<div style='color: {stability_color}; font-weight: 600; font-size: 0.9rem;'>{stability_level}</div>",
                        unsafe_allow_html=True)

                with col2:
                    avg_entropy = sum(data.get("entropy", 0) for data in stability_data.values()) / len(stability_data)
                    st.metric("Avg Rank Entropy", f"{avg_entropy:.3f}", help="Lower = more stable ranks")

                with col3:
                    st.metric("Simulation Samples", f"{samples_count:,}", help="Monte Carlo samples analyzed")

                # Weight variation summary
                if weight_stats:
                    st.markdown("#### Weight Variation Impact")

                    # Create weight variation summary table
                    weight_summary = []
                    for channel, stats in weight_stats.items():
                        variation_pct = (stats["std"] / stats["mean"]) * 100 if stats["mean"] > 0 else 0
                        weight_summary.append({
                            "Channel": channel.replace("_", " ").title(),
                            "Base Weight": f"{stats['base']:.3f}",
                            "Mean": f"{stats['mean']:.3f}",
                            "Std Dev": f"{stats['std']:.3f}",
                            "Variation %": f"{variation_pct:.1f}%"
                        })

                    weight_df = pd.DataFrame(weight_summary)
                    st.dataframe(weight_df, use_container_width=True, hide_index=True)

                # Per-target stability analysis
                st.markdown("#### Per-Target Rank Stability")

                # Sort targets by entropy (most unstable first)
                sorted_targets = sorted(
                    stability_data.items(),
                    key=lambda x: x[1].get("entropy", 0),
                    reverse=True
                )

                # Show top 6 targets in grid
                target_cols = st.columns(2)

                for i, (target, target_data) in enumerate(sorted_targets[:6]):
                    with target_cols[i % 2]:
                        mode_rank = target_data.get("mode_rank", 0)
                        entropy = target_data.get("entropy", 0)
                        rank_range = target_data.get("rank_range", [0, 0])
                        histogram = target_data.get("histogram", {})

                        # Create rank histogram visualization
                        if histogram:
                            hist_df = pd.DataFrame([
                                {"Rank": k, "Count": v} for k, v in histogram.items()
                            ]).sort_values("Rank")

                            # Color coding for stability
                            if entropy < 0.2:
                                border_color = "#34D399"  # Green - stable
                            elif entropy < 0.4:
                                border_color = "#22D3EE"  # Cyan - moderate
                            else:
                                border_color = "#F87171"  # Red - unstable

                            with st.container():
                                st.markdown(f"""
                                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: {VANTAI_THEME['bg_card']};">
                                    <div style="font-weight: 600; color: {VANTAI_THEME['text_primary']}; margin-bottom: 0.5rem;">{target}</div>
                                    <div style="color: {VANTAI_THEME['text_secondary']}; font-size: 0.85rem;">
                                        Mode Rank: <strong>{mode_rank}</strong> | Entropy: <strong>{entropy:.3f}</strong><br>
                                        Range: {rank_range[0]}-{rank_range[1]}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Simple bar chart for rank distribution
                                if len(hist_df) > 0:
                                    st.bar_chart(hist_df.set_index("Rank")["Count"], height=200)

                # Interpretation guide
                with st.expander("📊 How to interpret stability metrics"):
                    st.markdown("""
                    **Rank Correlation (Kendall's τ):**
                    - 0.9-1.0: Very stable rankings across weight perturbations
                    - 0.8-0.9: Stable with minor variations
                    - 0.6-0.8: Moderate stability, some rank changes
                    - <0.6: Unstable rankings, sensitive to weights

                    **Rank Entropy:**
                    - Low (0-0.3): Target consistently ranks in same position
                    - Medium (0.3-0.6): Some rank variation but generally stable
                    - High (0.6-1.0): High rank uncertainty across weight samples

                    **Mode Rank:** Most frequent rank across all weight samples

                    **Interpretation:** Targets with low entropy and tight rank ranges are robust 
                    to weight uncertainty, while high entropy targets may need more confident 
                    weight assignments or additional validation.
                    """)
            else:
                st.info("No stability data available from simulation")


# Add this CSS for enhanced styling (append to load_professional_css function)
STABILITY_CSS = """
/* Stability Analysis Specific Styles */
.stability-metric-card {
    background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.stability-target-card {
    background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.stability-target-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.1);
}

.stability-high { border-left: 4px solid #34D399; }
.stability-medium { border-left: 4px solid #22D3EE; }
.stability-low { border-left: 4px solid #F87171; }
"""

# Integration code for main() function:


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
    Render enhanced results table with pandas Styler, heatmap, and zebra rows.
    Falls back to standard dataframe if Styler fails.
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

    # Try to use pandas Styler for enhanced visualization
    try:
        # Define numeric columns for heatmap
        numeric_columns = ["Total Score", "Genetics", "PPI Network", "Pathway", "Safety", "Modality"]

        # Create styled dataframe
        styled_df = df.style

        # Apply background gradient (heatmap) to numeric columns
        for col in numeric_columns:
            if col in df.columns:
                if col == "Safety":
                    # For safety, lower is better, so reverse the colormap
                    styled_df = styled_df.background_gradient(
                        subset=[col],
                        cmap='RdYlGn',  # Red-Yellow-Green (reversed for safety)
                        vmin=df[col].min(),
                        vmax=df[col].max(),
                        text_color_threshold=0.5,
                        axis=0
                    )
                else:
                    # For other metrics, higher is better
                    styled_df = styled_df.background_gradient(
                        subset=[col],
                        cmap='RdYlGn_r',  # Green-Yellow-Red (normal)
                        vmin=df[col].min(),
                        vmax=df[col].max(),
                        text_color_threshold=0.5,
                        axis=0
                    )

        # Format numeric columns to 3 decimals with monospace font
        format_dict = {}
        for col in numeric_columns:
            if col in df.columns:
                format_dict[col] = lambda x: f"{x:.3f}"

        styled_df = styled_df.format(format_dict)

        # Apply zebra striping (alternating row colors)
        def zebra_stripe(row):
            """Apply zebra striping to rows."""
            if row.name % 2 == 0:
                return ['background-color: rgba(15, 23, 42, 0.5)'] * len(row)
            else:
                return ['background-color: rgba(30, 41, 59, 0.3)'] * len(row)

        styled_df = styled_df.apply(zebra_stripe, axis=1)

        # Apply monospace font to numeric columns
        def monospace_numeric(val, col_name):
            """Apply monospace font to numeric values."""
            if col_name in numeric_columns:
                return 'font-family: "SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace; font-variant-numeric: tabular-nums;'
            return ''

        # Apply custom CSS for better styling
        styled_df = styled_df.set_table_styles([
            # Header styling
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', '#1a1f2e'),
                    ('color', '#94a3b8'),
                    ('font-weight', '600'),
                    ('text-transform', 'uppercase'),
                    ('letter-spacing', '0.05em'),
                    ('font-size', '0.8rem'),
                    ('border-bottom', '1px solid #334155'),
                    ('padding', '0.75rem 1rem')
                ]
            },
            # Cell styling
            {
                'selector': 'tbody td',
                'props': [
                    ('color', '#e2e8f0'),
                    ('border-bottom', '1px solid #334155'),
                    ('padding', '0.75rem 1rem'),
                    ('font-size', '0.9rem')
                ]
            },
            # Table styling
            {
                'selector': 'table',
                'props': [
                    ('background-color', 'transparent'),
                    ('border-collapse', 'collapse'),
                    ('width', '100%'),
                    ('border-radius', '8px'),
                    ('overflow', 'hidden')
                ]
            },
            # Hover effect
            {
                'selector': 'tbody tr:hover',
                'props': [
                    ('background-color', 'rgba(34, 211, 238, 0.1) !important'),
                    ('transform', 'scale(1.01)'),
                    ('transition', 'all 0.2s ease')
                ]
            }
        ])

        # Apply monospace to numeric columns
        def apply_numeric_styles(styler):
            """Apply monospace styling to numeric columns."""
            for col in numeric_columns:
                if col in df.columns:
                    styler = styler.applymap(
                        lambda
                            x: 'font-family: "SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace; font-variant-numeric: tabular-nums; text-align: right;',
                        subset=[col]
                    )
            return styler

        styled_df = apply_numeric_styles(styled_df)

        # Display the styled dataframe
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=min(500, (len(df) + 1) * 45 + 40)  # Dynamic height
        )

    except Exception as e:
        # Fallback to standard dataframe if Styler fails
        st.warning(f"Enhanced styling unavailable, using standard table: {str(e)[:50]}...")

        # Standard column configuration as fallback
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

        # Display standard table
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


def render_actionable_explanation_panel_with_matrix(target_data: Dict, selected_target: str):
    """
    Enhanced explanation panel with evidence matrix integration.
    """
    if not target_data:
        st.info("No explanation data available for this target")
        return

    # Extract explanation data
    explanation = target_data.get("explanation", {}) or {}

    # Render evidence matrix first (if available)
    if explanation.get("evidence_refs"):
        st.markdown("#### Evidence Analysis")
        render_evidence_matrix(explanation)
        st.divider()  # Visual separator

    # Then render the existing explanation panel
    render_actionable_explanation_panel(target_data, selected_target)


def render_channel_ablation_analysis(results, last_request):
    """
    Render channel ablation analysis showing impact of removing each scoring channel.
    """
    with st.container():
        st.markdown('<div class="section-header">Channel Ablation Analysis</div>', unsafe_allow_html=True)
        st.caption("Analyze the impact of removing each scoring channel")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔬 Run Ablation Analysis", help="Remove each channel and measure impact on scores"):
                st.session_state["run_ablation"] = True

        with col2:
            st.markdown("""
            **Method:** Sets each channel weight to 0,  
            renormalizes others to sum=1.0
            """)

        # Run ablation if requested
        if st.session_state.get("run_ablation", False):
            with st.spinner("Running channel ablation analysis..."):
                try:
                    # Call ablation endpoint
                    response = call_api("/ablation", method="POST", data=last_request)

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

            # Extract data safely
            ablation_results = ablation_data.get("ablation_results", [])
            channel_criticality = ablation_data.get("channel_criticality", {})
            summary_stats = ablation_data.get("summary_stats", {})
            baseline_scores = ablation_data.get("baseline_scores", [])

            if ablation_results:
                # Overall channel criticality summary
                st.markdown("#### Channel Criticality Summary")

                col1, col2, col3 = st.columns(3)

                with col1:
                    most_critical = summary_stats.get("most_critical_channel", "Unknown")
                    st.metric("Most Critical", most_critical)

                    if most_critical in channel_criticality:
                        crit_level = channel_criticality[most_critical]["level"]
                        crit_impact = channel_criticality[most_critical]["avg_impact"]

                        color_map = {
                            "critical": "#F87171",
                            "important": "#F59E0B",
                            "minor": "#34D399"
                        }
                        color = color_map.get(crit_level, "#94A3B8")

                        st.markdown(f"""
                        <div style="color: {color}; font-weight: 600; font-size: 0.9rem;">
                            {crit_level.title()} (avg drop: {crit_impact:.3f})
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    least_critical = summary_stats.get("least_critical_channel", "Unknown")
                    st.metric("Least Critical", least_critical)

                    if least_critical in channel_criticality:
                        least_impact = channel_criticality[least_critical]["avg_impact"]
                        st.caption(f"avg drop: {least_impact:.3f}")

                with col3:
                    channels_analyzed = summary_stats.get("total_channels_analyzed", 0)
                    st.metric("Channels Analyzed", channels_analyzed)

                # Channel impact overview table
                st.markdown("#### Channel Impact Overview")

                channel_summary = []
                for ablation in ablation_results:
                    channel = ablation["channel"]
                    avg_drop = ablation["avg_score_drop"]
                    max_drop = ablation["max_score_drop"]
                    affected = ablation["targets_affected"]

                    crit_info = channel_criticality.get(channel, {})
                    criticality = crit_info.get("level", "unknown")

                    channel_summary.append({
                        "Channel": channel.replace("_", " ").title(),
                        "Avg Score Drop": f"{avg_drop:.3f}",
                        "Max Score Drop": f"{max_drop:.3f}",
                        "Targets Affected": affected,
                        "Criticality": criticality.title()
                    })

                channel_df = pd.DataFrame(channel_summary)
                st.dataframe(channel_df, use_container_width=True, hide_index=True)

                # Per-target ablation analysis
                st.markdown("#### Per-Target Impact Analysis")

                # Target selector for detailed analysis
                target_names = [item["target"] for item in baseline_scores]
                selected_target = st.selectbox(
                    "Select target for detailed ablation view",
                    target_names,
                    key="ablation_target_select"
                )

                if selected_target:
                    st.markdown(f"**Impact on {selected_target} when removing each channel:**")

                    # Collect data for selected target across all channels
                    target_impacts = []

                    for ablation in ablation_results:
                        channel = ablation["channel"]

                        # Find this target in the delta list
                        target_delta = next(
                            (delta for delta in ablation["delta"] if delta["target"] == selected_target),
                            None
                        )

                        if target_delta:
                            score_drop = target_delta["score_drop"]
                            rank_delta = target_delta["rank_delta"]

                            target_impacts.append({
                                "Channel": channel.replace("_", " ").title(),
                                "Score Drop": score_drop,
                                "Rank Change": rank_delta
                            })

                    # Sort by score drop (descending)
                    target_impacts.sort(key=lambda x: x["Score Drop"], reverse=True)

                    if target_impacts:
                        # Create bar chart data
                        chart_data = pd.DataFrame({
                            "Channel": [item["Channel"] for item in target_impacts],
                            "Score Drop": [item["Score Drop"] for item in target_impacts]
                        })

                        # Display bar chart
                        st.bar_chart(chart_data.set_index("Channel")["Score Drop"], height=300)

                        # Show detailed table
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Score Impact:**")
                            for impact in target_impacts:
                                channel = impact["Channel"]
                                drop = impact["Score Drop"]

                                # Determine if this is the most critical for this target
                                is_critical = (drop == max(item["Score Drop"] for item in target_impacts))

                                if is_critical and drop > 0.05:
                                    st.markdown(f"🔴 **{channel}**: -{drop:.3f} (critical)")
                                elif drop > 0.02:
                                    st.markdown(f"🟡 **{channel}**: -{drop:.3f}")
                                else:
                                    st.markdown(f"🟢 **{channel}**: -{drop:.3f}")

                        with col2:
                            st.markdown("**Rank Impact:**")
                            for impact in target_impacts:
                                channel = impact["Channel"]
                                rank_change = impact["Rank Change"]

                                if rank_change > 0:
                                    st.markdown(f"📉 **{channel}**: +{rank_change} (worse)")
                                elif rank_change < 0:
                                    st.markdown(f"📈 **{channel}**: {rank_change} (better)")
                                else:
                                    st.markdown(f"➡️ **{channel}**: no change")

                        # Identify most critical channel for this target
                        most_critical_impact = target_impacts[0] if target_impacts else None
                        if most_critical_impact and most_critical_impact["Score Drop"] > 0.05:
                            critical_channel = most_critical_impact["Channel"]
                            critical_drop = most_critical_impact["Score Drop"]

                            st.info(f"💡 **Critical channel for {selected_target}:** {critical_channel} "
                                    f"(removing it drops score by {critical_drop:.3f})")
                    else:
                        st.warning(f"No ablation data found for {selected_target}")

                # Interpretation guide
                with st.expander("📊 How to interpret ablation results"):
                    st.markdown("""
                    **Channel Criticality Levels:**
                    - 🔴 **Critical** (avg drop ≥ 0.15): Essential for accurate scoring
                    - 🟡 **Important** (avg drop ≥ 0.05): Significant contribution to scores
                    - 🟢 **Minor** (avg drop < 0.05): Limited impact on rankings

                    **Score Drop:** How much the total score decreases when a channel is removed

                    **Rank Change:** How position changes (positive = rank gets worse)

                    **Interpretation:** Channels with high score drops are critical for maintaining 
                    accurate target prioritization. Removing critical channels significantly 
                    changes rankings and may lead to suboptimal target selection.

                    **Use Case:** Use this analysis to:
                    - Identify which data sources are most valuable
                    - Understand scoring robustness
                    - Prioritize data quality improvements
                    - Validate weight assignments
                    """)
            else:
                st.info("No ablation data available from analysis")




import json
import os
from pathlib import Path
import numpy as np



import urllib.parse
import base64
import json


def encode_params_for_url(disease_id, targets, weights):
    """
    Encode parameters for URL sharing.
    """
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
    """
    Decode parameters from URL query params.
    """
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
    """
    Update URL query parameters with current state.
    """
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


def render_copy_link_button():
    """
    Render copy link button in sidebar.
    """
    try:
        # Get current URL
        current_url = st.runtime.caching.get_streamlit_runtime().get_client_state().browser_info.origin
        if not current_url:
            current_url = "http://localhost:8501"  # Fallback

        # Create shareable URL
        if hasattr(st, 'query_params') and st.query_params:
            query_string = urllib.parse.urlencode(dict(st.query_params))
            shareable_url = f"{current_url}?{query_string}"
        else:
            shareable_url = current_url

        st.markdown("**Share Analysis**")

        # Display URL in code block
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
        # Fallback: simple text copy
        st.markdown("**Share Analysis**")
        st.caption("Copy current URL to share this analysis configuration")


# Integration code for main() function
def load_state_from_url():
    """
    Load initial state from URL parameters.
    Returns: (disease_id, disease_name, targets, weights) or None values if not found
    """
    url_disease_id, url_targets, url_weights = decode_params_from_url()

    if url_disease_id:
        # Map disease ID back to name
        disease_options = {
            "Non-small cell lung carcinoma": "EFO_0000305",
            "Breast carcinoma": "EFO_0000305",
            "Colorectal carcinoma": "EFO_0000305"
        }

        # Find disease name by ID
        url_disease_name = None
        for name, id_val in disease_options.items():
            if id_val == url_disease_id:
                url_disease_name = name
                break

        if not url_disease_name:
            url_disease_name = list(disease_options.keys())[0]  # Default

        return url_disease_id, url_disease_name, url_targets, url_weights

    return None, None, None, None


# Modified sidebar section for main() function
def render_sidebar_with_url_state():
    """
    Render sidebar with URL state loading and sharing.
    """
    # Load state from URL
    url_disease_id, url_disease_name, url_targets, url_weights = load_state_from_url()

    # Disease selection
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Disease Context**")
    disease_options = {
        "Non-small cell lung carcinoma": "EFO_0000305",
        "Breast carcinoma": "EFO_0000305",
        "Colorectal carcinoma": "EFO_0000305"
    }

    # Use URL state if available
    default_disease = url_disease_name if url_disease_name else list(disease_options.keys())[0]
    selected_disease_name = st.selectbox(
        "Select Disease",
        list(disease_options.keys()),
        index=list(disease_options.keys()).index(default_disease) if default_disease in disease_options else 0,
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
        index=list(target_sets.keys()).index(initial_set) if initial_set in target_sets else 0
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
    st.markdown('</div>', unsafe_allow_html=True)

    # Scoring weights with URL defaults
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Algorithm Weights**")

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
    st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        render_copy_link_button()
        st.markdown('</div>', unsafe_allow_html=True)

    return selected_disease_name, disease_id, targets, weights
def create_demo_ground_truth():
    """
    Create demo ground truth data files in data_demo/ directory.
    """
    demo_dir = Path("data_demo")
    demo_dir.mkdir(exist_ok=True)

    # NSCLC ground truth
    nsclc_truth = {
        "disease": "Non-small cell lung carcinoma",
        "disease_id": "EFO_0000305",
        "positives": ["EGFR", "ALK", "MET", "ERBB2", "BRAF", "KRAS", "ROS1", "RET"],
        "negatives": ["GAPDH", "ACTB", "TUBB", "RPL13A", "HPRT1"],
        "description": "Known therapeutic targets (positives) vs housekeeping genes (negatives)",
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
    """
    Load ground truth data for disease benchmarking.
    """
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
        # Create demo data if it doesn't exist
        create_demo_ground_truth()

    try:
        with open(truth_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def compute_precision_at_k(scores: list, positives: set, k_values: list = [1, 3, 5]):
    """
    Compute precision@k metrics.

    Args:
        scores: List of (target, score) tuples sorted by score descending
        positives: Set of positive target identifiers
        k_values: List of k values to compute precision for

    Returns:
        Dict mapping k to precision@k
    """
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
    """
    Compute simple AUC-PR approximation using step function.

    Args:
        scores: List of (target, score) tuples sorted by score descending
        positives: Set of positive target identifiers

    Returns:
        Float AUC-PR approximation
    """
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
    """
    Render benchmark analysis panel comparing results to ground truth.
    """
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

    # Prepare data for benchmarking - handle both dict and object formats
    scored_targets = []
    for ts in target_scores:
        if hasattr(ts, 'target'):  # Object format
            target = ts.target
            score = ts.total_score
        else:  # Dictionary format
            target = ts.get('target', 'Unknown')
            score = ts.get('total_score', 0)
        scored_targets.append((target, score))

    scored_targets.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending

    positives = set(ground_truth["positives"])
    negatives = set(ground_truth["negatives"])

    # Filter to only targets that are in ground truth (positives or negatives)
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

    # Show which targets made the cut
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

    # Detailed breakdown
    st.markdown("#### Detailed Benchmark Results")
    st.markdown(f"**Ground Truth Source:** {ground_truth.get('source', 'Unknown')}")
    st.markdown(f"**Description:** {ground_truth.get('description', 'No description')}")

    # Create detailed results table
    benchmark_results = []
    for target, score in benchmark_targets:
        if target in positives:
            label = "Known Target"
            status = "✅"
        elif target in negatives:
            label = "Control Gene"
            status = "❌"
        else:
            label = "Unknown"
            status = "❓"

        benchmark_results.append({
            "Target": target,
            "Score": f"{score:.3f}",
            "Status": status,
            "Type": label
        })

    if benchmark_results:
        benchmark_df = pd.DataFrame(benchmark_results)
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    # Methodology note
    st.caption("""
        **Methodology:** Precision@k measures fraction of known targets in top k predictions. 
        AUC-PR summarizes precision-recall trade-off. Ground truth based on FDA-approved 
        therapies and clinical guidelines.
        """)




def estimate_channel_rank_impact(target_name, contributions, rank_impact_data):
    """
    Estimate per-channel rank impact based on contribution weights and actual rank changes.

    Args:
        target_name: Name of the target
        contributions: List of channel contributions from explanation
        rank_impact_data: Rank impact data from scoring results

    Returns:
        Dict mapping channel names to estimated rank deltas
    """
    if not rank_impact_data or not contributions:
        return {}

    # Find this target's rank impact
    target_rank_impact = next(
        (item for item in rank_impact_data if item.get("target") == target_name),
        None
    )

    if not target_rank_impact:
        return {}

    actual_delta = target_rank_impact.get("delta", 0)
    if actual_delta == 0:
        return {contrib["channel"]: 0 for contrib in contributions}

    # Calculate total contribution from available channels
    total_contribution = sum(contrib["contribution"] for contrib in contributions if contrib["available"])

    if total_contribution == 0:
        return {}

    # Estimate per-channel impact proportional to contribution
    channel_estimates = {}
    for contrib in contributions:
        channel = contrib["channel"]
        if contrib["available"] and contrib["contribution"] > 0:
            # Proportional allocation of total rank change
            proportion = contrib["contribution"] / total_contribution
            estimated_delta = int(round(actual_delta * proportion))
            channel_estimates[channel] = estimated_delta
        else:
            channel_estimates[channel] = 0

    return channel_estimates


def format_rank_delta_chip(delta, channel_name):
    """
    Format rank delta as a colored chip with tooltip.

    Args:
        delta: Estimated rank change (positive = rank got worse)
        channel_name: Name of the channel for tooltip

    Returns:
        HTML string for the chip
    """
    if delta == 0:
        return '<span style="color: #94A3B8; font-size: 0.8rem;" title="No estimated rank impact">→</span>'
    elif delta > 0:
        # Rank got worse (moved down)
        color = "#F87171"
        symbol = "↓"
        text = f"+{delta}"
        tooltip = f"Estimated rank impact: moved down {delta} positions due to {channel_name} weighting"
    else:
        # Rank got better (moved up)
        color = "#34D399"
        symbol = "↑"
        text = f"{delta}"  # Already negative
        tooltip = f"Estimated rank impact: moved up {abs(delta)} positions due to {channel_name} weighting"

    return f'''
    <span style="
        color: {color}; 
        font-size: 0.8rem; 
        font-weight: 600;
        margin-left: 0.5rem;
        padding: 0.1rem 0.3rem;
        background: {color}20;
        border-radius: 3px;
        border: 1px solid {color}40;
    " title="{tooltip}">
        {symbol}{abs(delta)}
    </span>
    '''


def render_enhanced_channel_contributions(target_name, contributions, rank_impact_data=None):
    """
    Render channel contributions with delta rank estimates.

    Args:
        target_name: Name of the target
        contributions: List of channel contributions
        rank_impact_data: Optional rank impact data for delta estimation
    """
    if not contributions:
        st.info("No channel contribution data available")
        return

    # Estimate per-channel rank impacts
    channel_deltas = estimate_channel_rank_impact(target_name, contributions, rank_impact_data)

    st.markdown("#### Channel Contributions")

    # Create contribution analysis
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

        # Get delta estimate
        delta_estimate = channel_deltas.get(channel, 0)
        delta_chip = format_rank_delta_chip(delta_estimate, channel) if rank_impact_data else ""

        # Create expandable section with delta chip in title
        title_with_delta = f"{display_name}: {contribution:.3f} (Weight: {weight:.2f}){delta_chip}"

        if available and score is not None:
            with st.expander(title_with_delta, expanded=True):
                # Progress bar showing contribution
                max_contribution = max([c["contribution"] for c in contributions]) if contributions else 1.0
                progress = contribution / max_contribution if max_contribution > 0 else 0
                st.progress(progress)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.metric("Raw Score", f"{score:.3f}")
                with col2:
                    st.metric("Weighted", f"{contribution:.3f}")

                # Show delta explanation if available
                if rank_impact_data and delta_estimate != 0:
                    if delta_estimate > 0:
                        st.info(f"Channel weighting estimated to move rank down by ~{delta_estimate} positions")
                    else:
                        st.info(f"Channel weighting estimated to move rank up by ~{abs(delta_estimate)} positions")

                # Add channel-specific interpretation
                interpretation = _get_channel_interpretation(channel, score)
                if interpretation:
                    st.info(interpretation)
        else:
            # Unavailable channel
            with st.expander(f"⚪ {display_name}: Not Available", expanded=False):
                st.caption("Data not available or score is zero for this channel")


# Integration function for the explanation panel
def render_actionable_explanation_panel_with_deltas(target_data: Dict, selected_target: str, rank_impact_data=None):
    """
    Enhanced explanation panel with delta rank estimates.
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

        # Channel contributions with delta estimates
        contributions = explanation.get("contributions", [])
        if contributions:
            render_enhanced_channel_contributions(selected_target, contributions, rank_impact_data)

        # Evidence references section (existing code)
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
# Integration code for main() function:




def filter_diagnostic_evidence(evidence_refs):
    """
    Filter evidence references into user-facing and diagnostic categories.

    Args:
        evidence_refs: List of evidence references

    Returns:
        Tuple of (user_evidence, diagnostic_evidence)
    """
    diagnostic_prefixes = [
        "RWR_", "Centrality_", "PPI_", "OT_cache", "STRING_cache",
        "Error_", "Debug_", "Internal_", "Cache_", "Fetch_", "API_",
        "Demo_", "Fallback_", "Timeout_", "Status_", "Version_"
    ]

    user_evidence = []
    diagnostic_evidence = []

    for ref in evidence_refs:
        # Handle both string and dict formats
        if isinstance(ref, dict):
            ref_text = ref.get("label", "")
        else:
            ref_text = str(ref)

        # Check if this is diagnostic evidence
        is_diagnostic = any(ref_text.startswith(prefix) for prefix in diagnostic_prefixes)

        if is_diagnostic:
            diagnostic_evidence.append(ref)
        else:
            user_evidence.append(ref)

    return user_evidence, diagnostic_evidence


def render_diagnostic_evidence_panel(diagnostic_evidence):
    """
    Render diagnostic evidence in a collapsible panel.

    Args:
        diagnostic_evidence: List of diagnostic evidence references
    """
    if not diagnostic_evidence:
        return

    with st.expander(f"🔧 Diagnostics ({len(diagnostic_evidence)} items)", expanded=False):
        st.markdown("**Technical Information & Debug Data**")
        st.caption("Internal system information for debugging and performance analysis")

        # Group diagnostic evidence by type
        diagnostic_groups = {
            "Network Analysis": [],
            "Data Fetching": [],
            "Caching": [],
            "Errors": [],
            "Other": []
        }

        for ref in diagnostic_evidence:
            ref_text = ref.get("label", str(ref)) if isinstance(ref, dict) else str(ref)

            if any(prefix in ref_text for prefix in ["RWR_", "Centrality_", "PPI_"]):
                diagnostic_groups["Network Analysis"].append(ref)
            elif any(prefix in ref_text for prefix in ["cache", "Cache_", "Fetch_", "API_"]):
                diagnostic_groups["Caching"].append(ref)
            elif any(prefix in ref_text for prefix in ["Error_", "Timeout_", "Status_"]):
                diagnostic_groups["Errors"].append(ref)
            elif any(prefix in ref_text for prefix in ["OT_", "STRING_"]):
                diagnostic_groups["Data Fetching"].append(ref)
            else:
                diagnostic_groups["Other"].append(ref)

        # Render each diagnostic group
        for group_name, group_refs in diagnostic_groups.items():
            if group_refs:
                st.markdown(f"**{group_name} ({len(group_refs)})**")

                # Create columns for better layout
                cols = st.columns(2)
                for i, ref in enumerate(group_refs):
                    with cols[i % 2]:
                        if isinstance(ref, dict):
                            label = ref.get("label", "Unknown")
                            url = ref.get("url", "#")

                            # Style based on diagnostic type
                            if "Error_" in label or "Timeout_" in label:
                                badge_color = "#F87171"  # Red for errors
                                icon = "❌"
                            elif "cache" in label.lower():
                                badge_color = "#34D399"  # Green for cache hits
                                icon = "💾"
                            elif "RWR_" in label or "Centrality_" in label:
                                badge_color = "#22D3EE"  # Cyan for network analysis
                                icon = "🕸️"
                            else:
                                badge_color = "#A78BFA"  # Purple for other
                                icon = "⚙️"

                            st.markdown(f"""
                            <div style="
                                background: {badge_color}15;
                                border: 1px solid {badge_color}40;
                                border-radius: 4px;
                                padding: 0.5rem;
                                margin: 0.25rem 0;
                                font-family: 'SF Mono', 'Monaco', monospace;
                                font-size: 0.8rem;
                                color: {badge_color};
                            ">
                                {icon} {label}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # String format
                            st.code(str(ref), language=None)


def render_evidence_section_with_diagnostics(explanation):
    """
    Render evidence section with separated user and diagnostic evidence.

    Args:
        explanation: Explanation object containing evidence_refs
    """
    evidence_refs = explanation.get("evidence_refs", [])
    if not evidence_refs:
        st.info("No evidence references available")
        return

    # Filter evidence into user and diagnostic categories
    user_evidence, diagnostic_evidence = filter_diagnostic_evidence(evidence_refs)

    # Render user-facing evidence
    if user_evidence:
        st.markdown("#### Supporting Evidence")

        # Use the existing tabbed evidence display for user evidence
        from dashboard.components.explanation_panel import render_evidence_badges_tabs

        # Create explanation subset with only user evidence
        user_explanation = {
            "evidence_refs": user_evidence
        }

        try:
            render_evidence_badges_tabs(user_explanation)
        except:
            # Fallback: simple list
            for ref in user_evidence:
                if isinstance(ref, dict):
                    label = ref.get("label", "Unknown")
                    url = ref.get("url", "#")
                    if url and url != "#":
                        st.markdown(f"[{label}]({url})")
                    else:
                        st.markdown(f"- {label}")
                else:
                    st.markdown(f"- {str(ref)}")

    # Render diagnostic evidence in separate panel
    if diagnostic_evidence:
        render_diagnostic_evidence_panel(diagnostic_evidence)


# Integration function to replace evidence section in explanation panel
def render_clean_explanation_panel(target_data: Dict, selected_target: str):
    """
    Render explanation panel with clean evidence separation.
    """
    if not target_data:
        st.info("No explanation data available for this target")
        return

    explanation = target_data.get("explanation", {}) or {}
    contributions = explanation.get("contributions", [])

    with st.container():
        st.markdown(f"### Why is {selected_target} ranked here?")
        st.caption("Analysis of scoring factors and supporting evidence")

        # Channel contributions (existing implementation)
        if contributions:
            st.markdown("#### Channel Contributions")
            for contrib in contributions:
                # ... existing contribution rendering code ...
                pass

        # Clean evidence section with diagnostics separation
        render_evidence_section_with_diagnostics(explanation)

        # Summary metrics (existing implementation)
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





def get_ppi_neighbors_data(target_gene, max_neighbors=10):
    """
    Get PPI neighbors data for a target gene.

    Args:
        target_gene: Gene symbol
        max_neighbors: Maximum number of neighbors to return

    Returns:
        Dict containing neighbors data or None if unavailable
    """
    try:
        # Try to import PPI network from the scoring modules
        from app.channels.ppi_proximity import ppi_network

        if not hasattr(ppi_network, 'graph') or target_gene not in ppi_network.graph:
            return None

        graph = ppi_network.graph

        # Get neighbors with edge weights
        neighbors = []
        for neighbor in graph.neighbors(target_gene):
            edge_data = graph.get_edge_data(target_gene, neighbor)
            weight = edge_data.get('weight', 0) if edge_data else 0
            neighbors.append({
                'gene': neighbor,
                'weight': float(weight),
                'edge_data': edge_data
            })

        # Sort by weight (descending) and limit
        neighbors.sort(key=lambda x: x['weight'], reverse=True)
        neighbors = neighbors[:max_neighbors]

        return {
            'target': target_gene,
            'neighbors': neighbors,
            'total_neighbors': len(list(graph.neighbors(target_gene))),
            'graph_available': True
        }

    except ImportError:
        return {'graph_available': False, 'error': 'PPI network not available'}
    except Exception as e:
        return {'graph_available': False, 'error': str(e)}


def check_interactive_network_viz():
    """
    Check if InteractiveNetworkViz is available.
    """
    try:
        from dashboard.components.network_viz import InteractiveNetworkViz
        return True, InteractiveNetworkViz
    except ImportError:
        return False, None


def render_mini_ppi_card(target_gene):
    """
    Render compact PPI network card for a target gene.

    Args:
        target_gene: Gene symbol to analyze
    """
    st.markdown("#### PPI Network Neighbors")

    # Get neighbors data
    ppi_data = get_ppi_neighbors_data(target_gene, max_neighbors=10)

    if not ppi_data or not ppi_data.get('graph_available'):
        error_msg = ppi_data.get('error', 'PPI network unavailable') if ppi_data else 'No PPI data found'
        st.info(f"PPI network analysis unavailable: {error_msg}")
        return

    neighbors = ppi_data.get('neighbors', [])
    total_neighbors = ppi_data.get('total_neighbors', 0)

    if not neighbors:
        st.info(f"No PPI neighbors found for {target_gene}")
        return

    # Check for interactive visualization
    viz_available, InteractiveNetworkViz = check_interactive_network_viz()

    # Header with statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target", target_gene)
    with col2:
        st.metric("Neighbors Shown", f"{len(neighbors)}")
    with col3:
        st.metric("Total Neighbors", f"{total_neighbors}")

    # Interactive network visualization (if available)
    if viz_available and len(neighbors) > 0:
        try:
            st.markdown("**Network Visualization**")

            # Create small subgraph centered on target
            viz = InteractiveNetworkViz()

            # Build subgraph data
            nodes = [{'id': target_gene, 'label': target_gene, 'color': '#22D3EE', 'size': 20}]
            edges = []

            for neighbor in neighbors:
                nodes.append({
                    'id': neighbor['gene'],
                    'label': neighbor['gene'],
                    'color': '#A78BFA',
                    'size': 15
                })
                edges.append({
                    'from': target_gene,
                    'to': neighbor['gene'],
                    'weight': neighbor['weight'],
                    'label': f"{neighbor['weight']:.3f}",
                    'color': '#94A3B8'
                })

            # Render compact network
            viz.render_network(
                nodes=nodes,
                edges=edges,
                height=300,
                title=f"PPI Network: {target_gene}"
            )

            # Full network link
            if st.button("🔍 Open Full Network Analysis", key=f"full_network_{target_gene}"):
                st.info("Full network analysis would open in expanded view")

        except Exception as e:
            st.warning(f"Network visualization failed: {str(e)[:50]}...")
            viz_available = False

    # Fallback: Neighbors table
    if not viz_available or len(neighbors) == 0:
        st.markdown("**First-Shell Neighbors**")

        if neighbors:
            # Create neighbors table
            neighbors_data = []
            for i, neighbor in enumerate(neighbors, 1):
                neighbors_data.append({
                    'Rank': i,
                    'Gene': neighbor['gene'],
                    'Weight': f"{neighbor['weight']:.4f}",
                    'Confidence': 'High' if neighbor['weight'] > 0.7 else 'Medium' if neighbor[
                                                                                          'weight'] > 0.4 else 'Low'
                })

            neighbors_df = pd.DataFrame(neighbors_data)

            # Display with custom styling
            st.dataframe(
                neighbors_df,
                use_container_width=True,
                hide_index=True,
                height=min(300, len(neighbors_df) * 35 + 40),
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Gene": st.column_config.TextColumn("Gene", width="medium"),
                    "Weight": st.column_config.TextColumn("Weight", width="small",
                                                          help="Edge weight from STRING database"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small")
                }
            )

            # Additional statistics
            if len(neighbors) > 0:
                avg_weight = sum(n['weight'] for n in neighbors) / len(neighbors)
                max_weight = max(n['weight'] for n in neighbors)

                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Average edge weight: {avg_weight:.3f}")
                with col2:
                    st.caption(f"Strongest connection: {max_weight:.3f}")

            # Show truncation notice
            if total_neighbors > len(neighbors):
                st.caption(f"Showing top {len(neighbors)} of {total_neighbors} total neighbors")
        else:
            st.info(f"No significant PPI neighbors found for {target_gene}")

    # Analysis insights
    if neighbors:
        with st.expander("Network Analysis Insights", expanded=False):
            high_confidence = [n for n in neighbors if n['weight'] > 0.7]
            medium_confidence = [n for n in neighbors if 0.4 <= n['weight'] <= 0.7]

            insights = []

            if high_confidence:
                insights.append(f"**Strong interactions:** {len(high_confidence)} high-confidence connections")
                top_partner = high_confidence[0]['gene']
                insights.append(f"**Primary partner:** {top_partner} (weight: {high_confidence[0]['weight']:.3f})")

            if medium_confidence:
                insights.append(f"**Moderate interactions:** {len(medium_confidence)} medium-confidence connections")

            if total_neighbors > 20:
                insights.append(f"**Hub protein:** {target_gene} has {total_neighbors} total connections (network hub)")
            elif total_neighbors > 10:
                insights.append(f"**Well-connected:** {target_gene} has {total_neighbors} connections")
            else:
                insights.append(f"**Peripheral:** {target_gene} has {total_neighbors} connections (network periphery)")

            for insight in insights:
                st.markdown(insight)


# Integration function for target details
def render_target_details_with_ppi(target_data, selected_target):
    """
    Render target details section with PPI network card.

    Args:
        target_data: Target data dictionary
        selected_target: Selected target gene name
    """
    # Existing target details rendering...

    # Add PPI network card
    render_mini_ppi_card(selected_target)

    # Continue with existing modality components, etc.


# Standalone PPI analysis function
def render_ppi_analysis_section():
    """
    Render standalone PPI analysis section (optional).
    """
    st.markdown("### PPI Network Analysis")

    # Gene input for ad-hoc analysis
    analysis_gene = st.text_input(
        "Analyze PPI neighbors for gene:",
        placeholder="Enter gene symbol (e.g., EGFR)",
        key="ppi_analysis_gene"
    )

    if analysis_gene and st.button("Analyze Network", key="analyze_ppi"):
        analysis_gene = analysis_gene.strip().upper()
        render_mini_ppi_card(analysis_gene)
# Main application
def main():
    """Main dashboard function with tabbed layout structure."""

    # Load professional theme with enhanced CSS
    load_professional_css_enhanced()

    # Enhanced Platform header
    st.markdown("""
    <div class="platform-header-enhanced">
        <div class="header-backdrop"></div>
        <div class="header-content">
            <div class="platform-title-large">VantAI Target Scoreboard</div>
            <div class="platform-subtitle-large">
                Advanced computational platform for modality-aware target prioritization using multi-omics integration
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
    selected_disease_name, disease_id, targets, weights = render_sidebar_with_url_state()

    # MAIN CONTENT AREA
    if "scoring_results" in st.session_state:
        results = st.session_state["scoring_results"]
        target_scores = results.get("targets", [])
        rank_impact = results.get("rank_impact", [])
        current_weights = st.session_state.get("last_request", {}).get("weights", {})

        if target_scores:
            # Configuration summary bar
            render_config_summary_bar(selected_disease_name, len(targets), weights)

            # TABBED LAYOUT - Main navigation
            tab_over, tab_rank, tab_explain, tab_ev, tab_sens, tab_bench = st.tabs([
                "📊 Overview", "🏆 Rankings", "🔍 Explain", "📚 Evidence", "⚖️ Sensitivity", "📈 Benchmark"
            ])

            with tab_over:
                render_analytics_overview(target_scores, results)

            with tab_rank:
                render_rankings_section(target_scores, rank_impact)

            with tab_explain:
                render_explain_section(target_scores, rank_impact, current_weights)

            with tab_ev:
                render_evidence_section(target_scores)

            with tab_sens:
                render_sensitivity_section(rank_impact, current_weights, target_scores)

            with tab_bench:
                render_benchmark_section(results, selected_disease_name)

            # Back to top button
            st.markdown('<a class="backtop" href="#">↑ Top</a>', unsafe_allow_html=True)

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


def load_professional_css_enhanced():
    """Enhanced CSS with improved typography and spacing."""
    enhanced_css = """
    <style>
    /* Base styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Enhanced Platform Header */
    .platform-header-enhanced {
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
    }

    .header-backdrop {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 30% 20%, rgba(34, 211, 238, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 70% 80%, rgba(167, 139, 250, 0.15) 0%, transparent 50%);
        pointer-events: none;
    }

    .header-content {
        position: relative;
        z-index: 1;
    }

    .platform-title-large {
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
    }

    .platform-subtitle-large {
        color: #94A3B8;
        font-size: 1.5rem;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
        letter-spacing: 0.01em;
    }

    .header-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 2rem;
    }

    .badge {
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
    }

    .badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(34, 211, 238, 0.4);
        border-color: #22D3EE;
    }

    /* Responsive design for header */
    @media (max-width: 768px) {
        .platform-header-enhanced {
            padding: 3rem 2rem;
        }
        .platform-title-large {
            font-size: 2.5rem;
        }
        .platform-subtitle-large {
            font-size: 1.2rem;
        }
        .header-badges {
            gap: 0.5rem;
        }
        .badge {
            font-size: 0.8rem;
            padding: 0.4rem 1rem;
        }
    }

    /* Enhanced typography */
    h2 { font-size: 1.6rem !important; margin: 1.75rem 0 .75rem 0 !important; }
    h3 { font-size: 1.25rem !important; margin: 1.25rem 0 .5rem 0 !important; }
    h4 { font-size: 1.05rem !important; color: #94A3B8 !important; }

    /* Section blocks */
    .section { 
        background: #0F172A; 
        border: 1px solid #1E293B; 
        border-radius: 12px; 
        padding: 1.25rem; 
        margin: 1rem 0 1.25rem; 
    }

    /* Config summary bar */
    .config-summary {
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
    }

    .config-item {
        color: #E2E8F0;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .config-value {
        color: #22D3EE;
        font-weight: 600;
    }

    /* Sticky navigation */
    .sticky-nav {
        position: sticky; 
        top: 0; 
        z-index: 999; 
        background: #0F172A;
        padding: .5rem 0; 
        border-bottom: 1px solid #1E293B;
        margin-bottom: 1rem;
    }

    .pill {
        display: inline-block; 
        margin: .25rem; 
        padding: .35rem .75rem; 
        border-radius: 999px;
        border: 1px solid #263247; 
        color: #E2E8F0; 
        font-size: .85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .pill.active {
        background: #22D3EE1a; 
        border-color: #22D3EE;
        color: #22D3EE;
    }

    .pill:hover {
        background: #1E293B;
        border-color: #475569;
    }

    /* Back to top */
    .backtop { 
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
    }

    .backtop:hover {
        background: #22D3EE20;
        border-color: #22D3EE;
        transform: translateY(-2px);
    }

    /* Enhanced metrics grid */
    .metrics-grid-enhanced {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-card-enhanced {
        background: linear-gradient(145deg, #0F172A 0%, #1A1F2E 100%);
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .metric-card-enhanced:hover {
        transform: translateY(-2px);
        border-color: #22D3EE40;
        box-shadow: 0 8px 24px rgba(34, 211, 238, 0.1);
    }
    </style>
    """
    st.markdown(enhanced_css, unsafe_allow_html=True)


def render_config_summary_bar(disease_name, target_count, weights):
    """Render configuration summary bar at the top."""
    top_weight = max(weights.items(), key=lambda x: x[1])
    weight_summary = f"{top_weight[0].title()}: {top_weight[1]:.2f}"

    st.markdown(f"""
    <div class="config-summary">
        <div class="config-item">Disease: <span class="config-value">{disease_name}</span></div>
        <div class="config-item">Targets: <span class="config-value">{target_count}</span></div>
        <div class="config-item">Top Weight: <span class="config-value">{weight_summary}</span></div>
        <div class="config-item">Analysis: <span class="config-value">Active</span></div>
    </div>
    """, unsafe_allow_html=True)


def sticky_local_nav(items, key="localnav"):
    """Render sticky local navigation within tabs."""
    if "local_section" not in st.session_state:
        st.session_state.local_section = items[0]

    with st.container():
        st.markdown('<div class="sticky-nav">', unsafe_allow_html=True)
        cols = st.columns(len(items))
        for i, item in enumerate(items):
            active_class = " active" if st.session_state.local_section == item else ""
            if cols[i].button(item, key=f"{key}_{item}"):
                st.session_state.local_section = item
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def render_analytics_overview(target_scores, results):
    """Render analytics overview section."""
    st.markdown("## Analytics Overview")

    total_scores = [ts.get("total_score", 0) for ts in target_scores]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Targets Analyzed", len(target_scores), help="Total targets processed")

    with col2:
        st.metric("Best Candidate", f"{max(total_scores):.3f}", help="Highest scoring target")

    with col3:
        st.metric("Mean Score", f"{sum(total_scores) / len(total_scores):.3f}", help="Cohort average")

    with col4:
        st.metric("Processing Time", f"{results.get('processing_time_ms', 0):.1f}ms", help="Computational efficiency")

    # Score distribution visualization
    st.markdown("### Score Distribution")
    score_df = pd.DataFrame({
        'Target': [ts.get('target', 'Unknown') for ts in target_scores],
        'Total Score': total_scores
    })
    st.bar_chart(score_df.set_index('Target')['Total Score'])


def render_rankings_section(target_scores, rank_impact):
    """Render rankings section with enhanced table."""
    st.markdown("## Target Rankings")

    sticky_local_nav(["Table", "Chart", "Comparison"], "rankings")

    if st.session_state.local_section == "Table":
        render_enhanced_results_table_with_progress(target_scores, rank_impact)
    elif st.session_state.local_section == "Chart":
        render_rankings_chart(target_scores)
    else:
        render_rankings_comparison(target_scores)


def render_enhanced_results_table_with_progress(target_scores, rank_impact=None):
    """Enhanced results table with progress columns."""
    if not target_scores:
        st.warning("No target scores to display")
        return

    sorted_targets = sorted(target_scores, key=lambda x: x.get("total_score", 0), reverse=True)
    rank_lookup = {}
    if rank_impact:
        rank_lookup = {item["target"]: item for item in rank_impact}

    table_data = []
    for i, ts in enumerate(sorted_targets, 1):
        target = ts.get('target', 'Unknown')
        breakdown = ts.get("breakdown", {})
        modality_fit = breakdown.get("modality_fit", {})

        rank_info = rank_lookup.get(target, {})
        movement = rank_info.get("movement", "unchanged")
        delta = rank_info.get("delta", 0)

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

    df = pd.DataFrame(table_data)

    # Enhanced column configuration with progress bars
    column_config = {
        "Rank": st.column_config.TextColumn("Rank", width="small"),
        "Target": st.column_config.TextColumn("Target", width="medium"),
        "Total Score": st.column_config.ProgressColumn("Total Score",
                                                       min_value=0.0, max_value=1.0, format="%.3f", width="medium"),
        "Genetics": st.column_config.ProgressColumn("Genetics",
                                                    min_value=0.0, max_value=1.0, format="%.3f"),
        "PPI Network": st.column_config.ProgressColumn("PPI",
                                                       min_value=0.0, max_value=1.0, format="%.3f"),
        "Pathway": st.column_config.ProgressColumn("Pathway",
                                                   min_value=0.0, max_value=1.0, format="%.3f"),
        "Safety": st.column_config.ProgressColumn("Safety (↓ better)",
                                                  min_value=0.0, max_value=1.0, format="%.3f"),
        "Modality": st.column_config.ProgressColumn("Modality",
                                                    min_value=0.0, max_value=1.0, format="%.3f"),
    }

    st.dataframe(
        df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(500, (len(df) + 1) * 40 + 40)
    )


def render_rankings_chart(target_scores):
    """Render rankings as horizontal bar chart."""
    sorted_targets = sorted(target_scores, key=lambda x: x.get("total_score", 0), reverse=True)[:10]

    chart_data = pd.DataFrame({
        'Target': [ts.get('target', 'Unknown') for ts in sorted_targets],
        'Score': [ts.get('total_score', 0) for ts in sorted_targets]
    })

    st.bar_chart(chart_data.set_index('Target')['Score'], horizontal=True)


def render_rankings_comparison(target_scores):
    """Render side-by-side target comparison."""
    st.markdown("### Target Comparison")

    if len(target_scores) >= 2:
        col1, col2 = st.columns(2)
        target_names = [ts.get('target', 'Unknown') for ts in target_scores]

        with col1:
            target1 = st.selectbox("First Target", target_names, key="compare1")
        with col2:
            target2 = st.selectbox("Second Target", target_names, key="compare2")

        if target1 != target2:
            data1 = next(ts for ts in target_scores if ts.get('target') == target1)
            data2 = next(ts for ts in target_scores if ts.get('target') == target2)

            comparison_data = {
                'Metric': ['Total Score', 'Genetics', 'PPI', 'Pathway', 'Safety', 'Modality'],
                target1: [
                    data1.get('total_score', 0),
                    data1.get('breakdown', {}).get('genetics', 0) or 0,
                    data1.get('breakdown', {}).get('ppi_proximity', 0) or 0,
                    data1.get('breakdown', {}).get('pathway_enrichment', 0) or 0,
                    data1.get('breakdown', {}).get('safety_off_tissue', 0) or 0,
                    (data1.get('breakdown', {}).get('modality_fit', {}) or {}).get('overall_druggability', 0) or 0
                ],
                target2: [
                    data2.get('total_score', 0),
                    data2.get('breakdown', {}).get('genetics', 0) or 0,
                    data2.get('breakdown', {}).get('ppi_proximity', 0) or 0,
                    data2.get('breakdown', {}).get('pathway_enrichment', 0) or 0,
                    data2.get('breakdown', {}).get('safety_off_tissue', 0) or 0,
                    (data2.get('breakdown', {}).get('modality_fit', {}) or {}).get('overall_druggability', 0) or 0
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_explain_section(target_scores, rank_impact, current_weights):
    """Render explanation section."""
    st.markdown("## Target Explanation")

    target_names = [ts.get("target", "Unknown") for ts in target_scores]
    selected_target = st.selectbox("Select target for detailed analysis", target_names, key="explain_target")

    if selected_target:
        target_data = next((ts for ts in target_scores if ts.get("target") == selected_target), None)
        if target_data:
            sticky_local_nav(["Contributions", "Network", "Modality"], "explain")

            if st.session_state.local_section == "Contributions":
                try:
                    render_actionable_explanation_panel_with_deltas(target_data, selected_target, rank_impact)
                except:
                    render_actionable_explanation_panel(target_data, selected_target)

            elif st.session_state.local_section == "Network":
                try:
                    render_mini_ppi_card(selected_target)
                except:
                    st.info("PPI network analysis unavailable")

            else:  # Modality
                render_modality_components(target_data)


def render_modality_components(target_data):
    """Render modality components section."""
    modality_fit = (target_data.get("breakdown", {}) or {}).get("modality_fit", {}) or {}

    if modality_fit:
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
    else:
        st.info("No modality fit data available for this target")


def render_evidence_section(target_scores):
    """Render evidence section with matrix and filters."""
    st.markdown("## Supporting Evidence")

    sticky_local_nav(["Matrix", "Details", "Diagnostics"], "evidence")

    if st.session_state.local_section == "Matrix":
        render_supporting_evidence_matrix(target_scores)
    elif st.session_state.local_section == "Details":
        render_evidence_details(target_scores)
    else:
        render_evidence_diagnostics(target_scores)


def render_supporting_evidence_matrix(target_scores):
    """Render evidence distribution matrix."""
    st.markdown("### Evidence Distribution")

    types = {
        "literature": "📚 Literature",
        "database": "🗄️ Databases",
        "proprietary": "🔬 VantAI",
        "other": "🔎 Other"
    }

    counts = {k: 0 for k in types}

    for ts in target_scores:
        evidence_refs = (ts.get("explanation") or {}).get("evidence_refs", [])
        for ref in evidence_refs:
            ref_type = ref.get("type", "other") if isinstance(ref, dict) else "other"
            counts[ref_type] = counts.get(ref_type, 0) + 1

    cols = st.columns(len(types))
    for i, (k, label) in enumerate(types.items()):
        cols[i].metric(label, counts.get(k, 0))

    # Filter by evidence types
    selected_types = st.multiselect(
        "Filter evidence types",
        options=list(types.keys()),
        default=list(types.keys()),
        format_func=lambda x: types[x]
    )

    # Display filtered evidence
    if selected_types:
        for ts in target_scores:
            target_name = ts.get("target", "Unknown")
            evidence_refs = (ts.get("explanation") or {}).get("evidence_refs", [])

            filtered_evidence = [
                ref for ref in evidence_refs
                if (ref.get("type", "other") if isinstance(ref, dict) else "other") in selected_types
            ]

            if filtered_evidence:
                with st.expander(f"{target_name} ({len(filtered_evidence)} evidence)"):
                    for ref in filtered_evidence:
                        if isinstance(ref, dict):
                            label = ref.get("label", "Unknown")
                            url = ref.get("url", "#")
                            if url and url != "#":
                                st.markdown(f"[{label}]({url})")
                            else:
                                st.markdown(f"- {label}")
                        else:
                            st.markdown(f"- {str(ref)}")


def render_evidence_details(target_scores):
    """Render detailed evidence for selected target."""
    target_names = [ts.get("target", "Unknown") for ts in target_scores]
    selected_target = st.selectbox("Select target for evidence details", target_names, key="evidence_target")

    if selected_target:
        target_data = next((ts for ts in target_scores if ts.get("target") == selected_target), None)
        if target_data:
            explanation = target_data.get("explanation", {})
            if explanation:
                try:
                    render_evidence_section_with_diagnostics(explanation)
                except:
                    render_evidence_matrix(explanation)


def render_evidence_diagnostics(target_scores):
    """Render diagnostic evidence information."""
    st.markdown("### Diagnostic Information")

    all_diagnostics = []
    for ts in target_scores:
        evidence_refs = (ts.get("explanation") or {}).get("evidence_refs", [])
        user_evidence, diagnostic_evidence = filter_diagnostic_evidence(evidence_refs)
        if diagnostic_evidence:
            all_diagnostics.extend(diagnostic_evidence)

    if all_diagnostics:
        render_diagnostic_evidence_panel(all_diagnostics)
    else:
        st.info("No diagnostic information available")


def render_sensitivity_section(rank_impact, current_weights, target_scores):
    """Render sensitivity analysis section."""
    st.markdown("## Sensitivity Analysis")

    sticky_local_nav(["Weight Impact", "Ablation", "Stability"], "sensitivity")

    if st.session_state.local_section == "Weight Impact":
        if rank_impact:
            try:
                render_ranking_impact_analysis(rank_impact, current_weights)
            except:
                st.info("Weight impact analysis unavailable")
        else:
            st.info("No ranking impact data available")

    elif st.session_state.local_section == "Ablation":
        try:
            render_channel_ablation_analysis(
                {"targets": target_scores},
                {"weights": current_weights}
            )
        except:
            st.info("Ablation analysis unavailable")

    else:  # Stability
        try:
            render_stability_sensitivity_analysis(
                {"targets": target_scores},
                {"weights": current_weights}
            )
        except:
            st.info("Stability analysis unavailable")


def render_benchmark_section(results, selected_disease_name):
    """Render benchmark analysis section."""
    st.markdown("## Benchmark Analysis")

    try:
        render_benchmark_panel(results, selected_disease_name)
    except:
        st.info("Benchmark analysis unavailable")


# Helper functions that need to be defined or imported
def filter_diagnostic_evidence(evidence_refs):
    """Filter evidence into user and diagnostic categories."""
    diagnostic_prefixes = [
        "RWR_", "Centrality_", "PPI_", "OT_cache", "STRING_cache",
        "Error_", "Debug_", "Internal_", "Cache_", "Fetch_", "API_",
        "Demo_", "Fallback_", "Timeout_", "Status_", "Version_"
    ]

    user_evidence = []
    diagnostic_evidence = []

    for ref in evidence_refs:
        ref_text = ref.get("label", str(ref)) if isinstance(ref, dict) else str(ref)
        is_diagnostic = any(ref_text.startswith(prefix) for prefix in diagnostic_prefixes)

        if is_diagnostic:
            diagnostic_evidence.append(ref)
        else:
            user_evidence.append(ref)

    return user_evidence, diagnostic_evidence


def render_diagnostic_evidence_panel(diagnostic_evidence):
    """Render diagnostic evidence panel."""
    if not diagnostic_evidence:
        return

    st.markdown("**Technical Information & Debug Data**")
    st.caption("Internal system information for debugging and performance analysis")

    for ref in diagnostic_evidence:
        ref_text = ref.get("label", str(ref)) if isinstance(ref, dict) else str(ref)
        st.code(ref_text, language=None)
if __name__ == "__main__":
    main()