"""
Evidence Matrix Component for dashboard/components/explanation_panel.py
Create this new file or add to existing explanation panel module.
"""

import streamlit as st
from typing import Dict, List
import re


def render_evidence_matrix(explanation: Dict):
    """
    Render evidence matrix grid showing evidence distribution across channels.

    Analyzes evidence_refs by type/tag and displays in a 5-column grid
    corresponding to scoring channels (Genetics, PPI, Pathway, Safety, Modality).

    Args:
        explanation: Explanation dictionary containing evidence_refs
    """
    if not explanation or "evidence_refs" not in explanation:
        st.info("No evidence references available for matrix analysis")
        return

    evidence_refs = explanation.get("evidence_refs", [])
    if not evidence_refs:
        st.info("No evidence references found")
        return

    # Define channel mapping
    channel_mapping = {
        "genetics": {
            "name": "Genetics",
            "icon": "🧬",
            "keywords": ["opentargets", "ot-", "genetics", "gwas", "variant", "association"]
        },
        "ppi": {
            "name": "PPI Network",
            "icon": "🕸️",
            "keywords": ["string", "ppi", "interaction", "network", "protein"]
        },
        "pathway": {
            "name": "Pathway",
            "icon": "🔬",
            "keywords": ["reactome", "pathway", "kegg", "go:", "biological"]
        },
        "safety": {
            "name": "Safety",
            "icon": "⚠️",
            "keywords": ["safety", "tissue", "expression", "toxicity", "side"]
        },
        "modality": {
            "name": "Modality",
            "icon": "💊",
            "keywords": ["vantai", "modality", "druggability", "protac", "degrader", "e3"]
        }
    }

    # Initialize evidence counters
    channel_evidence = {}
    for channel_id, channel_info in channel_mapping.items():
        channel_evidence[channel_id] = {
            "total_badges": 0,
            "external_links": 0,
            "latest_version": None,
            "evidence_types": set(),
            "raw_refs": []
        }

    # Analyze evidence references
    for ref in evidence_refs:
        # Handle both string and dict evidence formats
        if isinstance(ref, dict):
            ref_text = ref.get("label", "").lower()
            ref_url = ref.get("url", "")
            ref_type = ref.get("type", "unknown")
        else:
            ref_text = str(ref).lower()
            ref_url = ""
            ref_type = "unknown"

        # Classify evidence by channel
        classified = False

        for channel_id, channel_info in channel_mapping.items():
            keywords = channel_info["keywords"]

            # Check if reference matches this channel
            if any(keyword in ref_text for keyword in keywords):
                channel_data = channel_evidence[channel_id]
                channel_data["total_badges"] += 1
                channel_data["raw_refs"].append(ref)

                # Count external links
                if ref_url and ref_url not in ["#", ""]:
                    channel_data["external_links"] += 1

                # Track evidence types
                if isinstance(ref, dict):
                    channel_data["evidence_types"].add(ref_type)

                # Extract version information
                version_match = re.search(r'(\d{4}[\.\-]\d{2}|\d{4}|v\d+[\.\d]*)', ref_text)
                if version_match:
                    version = version_match.group(1)
                    if not channel_data["latest_version"] or version > channel_data["latest_version"]:
                        channel_data["latest_version"] = version

                classified = True
                break


"""
Evidence Matrix Component for dashboard/components/explanation_panel.py
Create this new file or add to existing explanation panel module.
"""

import streamlit as st
from typing import Dict, List
import re


def render_evidence_badges_tabs(explanation: Dict):
    """
    Render evidence badges in Streamlit tabs grouped by type with search filters.

    Args:
        explanation: Explanation dictionary containing evidence_refs
    """
    if not explanation or "evidence_refs" not in explanation:
        st.info("No evidence references available")
        return

    evidence_refs = explanation.get("evidence_refs", [])
    if not evidence_refs:
        st.info("No evidence references found")
        return

    # Group evidence by type
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
            # Convert string references to dict format
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

    # Literature tab
    with tabs[0]:
        literature_refs = evidence_by_type.get("literature", [])
        _render_evidence_tab(literature_refs, "literature")

    # Databases tab
    with tabs[1]:
        database_refs = evidence_by_type.get("database", [])
        _render_evidence_tab(database_refs, "database")

    # VantAI tab
    with tabs[2]:
        proprietary_refs = evidence_by_type.get("proprietary", [])
        _render_evidence_tab(proprietary_refs, "proprietary")

    # Other tab
    with tabs[3]:
        other_refs = evidence_by_type.get("other", [])
        _render_evidence_tab(other_refs, "other")


def _render_evidence_tab(refs: List[Dict], tab_type: str):
    """
    Render evidence badges for a specific tab with search functionality.

    Args:
        refs: List of evidence reference dictionaries
        tab_type: Type of tab for unique keys
    """
    if not refs:
        st.info(f"No {tab_type} evidence available")
        return

    # Search filter
    search_key = f"{tab_type}_search"
    search_term = st.text_input(
        f"Search {tab_type} evidence:",
        key=search_key,
        placeholder="Filter by label..."
    )

    # Filter references based on search
    filtered_refs = refs
    if search_term:
        filtered_refs = [
            ref for ref in refs
            if search_term.lower() in ref.get("label", "").lower()
        ]

    if not filtered_refs:
        st.warning(f"No {tab_type} evidence matches '{search_term}'")
        return

    # Display badge count after filtering
    if search_term:
        st.caption(f"Showing {len(filtered_refs)} of {len(refs)} references")

    # Render badges in columns for better layout
    cols_per_row = 3
    for i in range(0, len(filtered_refs), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, ref in enumerate(filtered_refs[i:i + cols_per_row]):
            with cols[j]:
                label = ref.get("label", "Unknown")
                url = ref.get("url", "#")

                # Create badge styling based on availability
                if url and url != "#":
                    # External link - create link button
                    st.markdown(f"""
                    <a href="{url}" target="_blank" style="
                        display: inline-block;
                        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
                        color: #22D3EE;
                        padding: 0.5rem 0.75rem;
                        border-radius: 6px;
                        text-decoration: none;
                        font-size: 0.85rem;
                        font-weight: 500;
                        margin: 0.25rem 0;
                        border: 1px solid #22D3EE40;
                        width: 100%;
                        text-align: center;
                        transition: all 0.2s ease;
                    " onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 4px 12px rgba(34, 211, 238, 0.3)';" 
                       onmouseout="this.style.transform=''; this.style.boxShadow='';">
                        {label}
                    </a>
                    """, unsafe_allow_html=True)
                else:
                    # Internal/unavailable - disabled style
                    st.markdown(f"""
                    <div style="
                        display: inline-block;
                        background: linear-gradient(145deg, #374151 0%, #4B5563 100%);
                        color: #9CA3AF;
                        padding: 0.5rem 0.75rem;
                        border-radius: 6px;
                        font-size: 0.85rem;
                        font-weight: 500;
                        margin: 0.25rem 0;
                        border: 1px solid #6B728040;
                        width: 100%;
                        text-align: center;
                        opacity: 0.7;
                    ">
                        {label}
                    </div>
                    """, unsafe_allow_html=True)


def render_evidence_matrix(explanation: Dict):
    """
    Render evidence matrix grid showing evidence distribution across channels.

    Analyzes evidence_refs by type/tag and displays in a 5-column grid
    corresponding to scoring channels (Genetics, PPI, Pathway, Safety, Modality).

    Args:
        explanation: Explanation dictionary containing evidence_refs
    """
    if not explanation or "evidence_refs" not in explanation:
        st.info("No evidence references available for matrix analysis")
        return

    evidence_refs = explanation.get("evidence_refs", [])
    if not evidence_refs:
        st.info("No evidence references found")
        return

    # Define channel mapping
    channel_mapping = {
        "genetics": {
            "name": "Genetics",
            "icon": "🧬",
            "keywords": ["opentargets", "ot-", "genetics", "gwas", "variant", "association"]
        },
        "ppi": {
            "name": "PPI Network",
            "icon": "🕸️",
            "keywords": ["string", "ppi", "interaction", "network", "protein"]
        },
        "pathway": {
            "name": "Pathway",
            "icon": "🔬",
            "keywords": ["reactome", "pathway", "kegg", "go:", "biological"]
        },
        "safety": {
            "name": "Safety",
            "icon": "⚠️",
            "keywords": ["safety", "tissue", "expression", "toxicity", "side"]
        },
        "modality": {
            "name": "Modality",
            "icon": "💊",
            "keywords": ["vantai", "modality", "druggability", "protac", "degrader", "e3"]
        }
    }

    # Initialize evidence counters
    channel_evidence = {}
    for channel_id, channel_info in channel_mapping.items():
        channel_evidence[channel_id] = {
            "total_badges": 0,
            "external_links": 0,
            "latest_version": None,
            "evidence_types": set(),
            "raw_refs": []
        }

    # Analyze evidence references
    for ref in evidence_refs:
        # Handle both string and dict evidence formats
        if isinstance(ref, dict):
            ref_text = ref.get("label", "").lower()
            ref_url = ref.get("url", "")
            ref_type = ref.get("type", "unknown")
        else:
            ref_text = str(ref).lower()
            ref_url = ""
            ref_type = "unknown"

        # Classify evidence by channel
        classified = False

        for channel_id, channel_info in channel_mapping.items():
            keywords = channel_info["keywords"]

            # Check if reference matches this channel
            if any(keyword in ref_text for keyword in keywords):
                channel_data = channel_evidence[channel_id]
                channel_data["total_badges"] += 1
                channel_data["raw_refs"].append(ref)

                # Count external links
                if ref_url and ref_url not in ["#", ""]:
                    channel_data["external_links"] += 1

                # Track evidence types
                if isinstance(ref, dict):
                    channel_data["evidence_types"].add(ref_type)

                # Extract version information
                version_match = re.search(r'(\d{4}[\.\-]\d{2}|\d{4}|v\d+[\.\d]*)', ref_text)
                if version_match:
                    version = version_match.group(1)
                    if not channel_data["latest_version"] or version > channel_data["latest_version"]:
                        channel_data["latest_version"] = version

                classified = True
                break

        # Handle unclassified evidence (assign to genetics as default)
        if not classified:
            channel_evidence["genetics"]["total_badges"] += 1
            channel_evidence["genetics"]["raw_refs"].append(ref)

    # Render evidence matrix
    st.markdown("#### Evidence Distribution Matrix")

    # Create 5-column layout
    cols = st.columns(5)

    for i, (channel_id, channel_info) in enumerate(channel_mapping.items()):
        with cols[i]:
            channel_data = channel_evidence[channel_id]
            channel_name = channel_info["name"]
            channel_icon = channel_info["icon"]

            # Determine card styling based on evidence count
            total_badges = channel_data["total_badges"]
            external_links = channel_data["external_links"]

            if total_badges >= 3:
                border_color = "#34D399"  # Green - well supported
                bg_gradient = "linear-gradient(145deg, #064e3b 0%, #065f46 100%)"
            elif total_badges >= 1:
                border_color = "#22D3EE"  # Cyan - some evidence
                bg_gradient = "linear-gradient(145deg, #164e63 0%, #0e7490 100%)"
            else:
                border_color = "#6B7280"  # Gray - no evidence
                bg_gradient = "linear-gradient(145deg, #374151 0%, #4B5563 100%)"

            # Create evidence card
            st.markdown(f"""
            <div style="
                background: {bg_gradient};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin-bottom: 1rem;
                min-height: 120px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{channel_icon}</div>
                    <div style="font-weight: 600; color: #E2E8F0; font-size: 0.9rem; margin-bottom: 0.5rem;">
                        {channel_name}
                    </div>
                </div>
                <div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">
                        <div><strong>{total_badges}</strong> badges</div>
                        <div><strong>{external_links}</strong> ext links</div>
                        {f'<div>v{channel_data["latest_version"]}</div>' if channel_data["latest_version"] else '<div>no version</div>'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show evidence types if available
            if channel_data["evidence_types"]:
                types_text = ", ".join(sorted(channel_data["evidence_types"]))
                st.caption(f"Types: {types_text}")

    # Summary statistics
    st.markdown("#### Evidence Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_evidence = sum(data["total_badges"] for data in channel_evidence.values())
        st.metric("Total Evidence", total_evidence)

    with col2:
        total_external = sum(data["external_links"] for data in channel_evidence.values())
        st.metric("External Links", total_external)

    with col3:
        channels_with_evidence = sum(1 for data in channel_evidence.values() if data["total_badges"] > 0)
        st.metric("Channels Covered", f"{channels_with_evidence}/5")

    # Render evidence badges in tabs
    st.markdown("#### Evidence References")
    render_evidence_badges_tabs(explanation)


def classify_evidence_by_database(evidence_refs: List) -> Dict[str, List]:
    """
    Classify evidence references by database/source type.

    Args:
        evidence_refs: List of evidence references (strings or dicts)

    Returns:
        Dict mapping database names to lists of references
    """
    databases = {
        "Open Targets": [],
        "STRING": [],
        "Reactome": [],
        "PubMed": [],
        "VantAI": [],
        "Other": []
    }

    for ref in evidence_refs:
        ref_text = ref.get("label", str(ref)) if isinstance(ref, dict) else str(ref)
        ref_lower = ref_text.lower()

        if "opentargets" in ref_lower or "ot-" in ref_lower:
            databases["Open Targets"].append(ref)
        elif "string" in ref_lower:
            databases["STRING"].append(ref)
        elif "reactome" in ref_lower:
            databases["Reactome"].append(ref)
        elif "pmid" in ref_lower:
            databases["PubMed"].append(ref)
        elif "vantai" in ref_lower:
            databases["VantAI"].append(ref)
        else:
            databases["Other"].append(ref)

    return databases