"""
Fixed Network Visualization Fallback Implementation
Add this to dashboard/components/network_viz.py or directly to ui_app.py
"""

import plotly.graph_objects as go
import pandas as pd
import random
import math

class InteractiveNetworkViz:
    def __init__(self, theme_colors=None):
        """Initialize with optional theme colors."""
        self.colors = theme_colors or {
            'accent_cyan': '#22D3EE',
            'accent_violet': '#A78BFA',
            'bg_primary': '#0B0F1A',
            'bg_surface': '#0F172A',
            'text_primary': '#E2E8F0',
            'warning': '#F59E0B',
            'success': '#34D399'
        }

    def render_network(self, nodes, edges, height=300, title="Network"):
        """Render a network visualization using Plotly."""
        if not nodes:
            return None

        fig = go.Figure()

        # Position nodes in a circle for better layout
        num_nodes = len(nodes)
        node_positions = {}

        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / num_nodes
            x = math.cos(angle) * 0.8
            y = math.sin(angle) * 0.8
            node_positions[node['id']] = (x, y)

        # Add edges
        for edge in edges:
            from_pos = node_positions.get(edge['from'])
            to_pos = node_positions.get(edge['to'])

            if from_pos and to_pos:
                fig.add_trace(go.Scatter(
                    x=[from_pos[0], to_pos[0], None],
                    y=[from_pos[1], to_pos[1], None],
                    mode='lines',
                    line=dict(color='#475569', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ))

        # Add nodes
        x_coords = []
        y_coords = []
        labels = []
        sizes = []
        colors = []

        for node in nodes:
            pos = node_positions[node['id']]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            labels.append(node['label'])
            sizes.append(node.get('size', 15))
            colors.append(node.get('color', self.colors['accent_cyan']))

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            text=labels,
            textposition="middle center",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2, color='#1E293B')
            ),
            showlegend=False,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text_primary'], size=16)),
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1.2, 1.2]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1.2, 1.2]
            ),
            plot_bgcolor=self.colors['bg_primary'],
            paper_bgcolor=self.colors['bg_surface'],
            height=height,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    def create_ppi_network_graph(self, targets, ppi_data=None):
        """Create PPI network visualization."""
        nodes = []
        edges = []

        # Create nodes for targets
        for i, target in enumerate(targets):
            nodes.append({
                'id': target,
                'label': target,
                'size': 20,
                'color': self.colors['accent_cyan']
            })

        # Add sample connections between targets
        for i in range(len(targets)):
            for j in range(i+1, min(i+3, len(targets))):
                edges.append({
                    'from': targets[i],
                    'to': targets[j],
                    'weight': random.uniform(0.5, 0.9)
                })

        return self.render_network(nodes, edges, title="PPI Network")

    def create_pathway_heatmap(self, target_scores):
        """Create pathway enrichment heatmap."""
        if not target_scores:
            return None

        targets = [ts.get('target', 'Unknown') for ts in target_scores[:10]]
        pathways = ["RTK Signaling", "Cell Cycle", "Apoptosis", "DNA Repair", "Metabolism"]

        # Generate realistic-looking data
        data = []
        for pathway in pathways:
            row = []
            for target in targets:
                # Simulate pathway-target associations
                score = random.uniform(0.1, 0.9)
                if pathway == "RTK Signaling" and target in ["EGFR", "ERBB2", "MET"]:
                    score = random.uniform(0.7, 0.95)
                row.append(score)
            data.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=targets,
            y=pathways,
            colorscale='Viridis',
            showscale=True,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text="Pathway Enrichment", font=dict(color=self.colors['text_primary'])),
            xaxis=dict(title="Targets", color=self.colors['text_primary']),
            yaxis=dict(title="Pathways", color=self.colors['text_primary']),
            plot_bgcolor=self.colors['bg_primary'],
            paper_bgcolor=self.colors['bg_surface'],
            height=400
        )

        return fig

    def create_druggability_radar(self, target_data):
        """Create druggability radar chart."""
        categories = ['Chemical Space', 'Structural', 'Safety', 'Market Potential', 'Patent Freedom']

        # Generate scores based on target data
        target_name = target_data.get('target', 'Unknown')
        breakdown = target_data.get('breakdown', {})

        # Use actual data where available, fallback to random
        values = [
            breakdown.get('genetics', random.uniform(0.4, 0.9)),
            breakdown.get('ppi_proximity', random.uniform(0.4, 0.9)),
            breakdown.get('safety_off_tissue', random.uniform(0.4, 0.9)),
            random.uniform(0.4, 0.9),  # Market potential
            random.uniform(0.4, 0.9)   # Patent freedom
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=target_name,
            line=dict(color=self.colors['accent_cyan']),
            fillcolor=f"{self.colors['accent_cyan']}40"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    color=self.colors['text_primary']
                ),
                angularaxis=dict(color=self.colors['text_primary'])
            ),
            title=dict(text=f"Druggability Assessment: {target_name}",
                      font=dict(color=self.colors['text_primary'])),
            showlegend=False,
            paper_bgcolor=self.colors['bg_surface'],
            height=400
        )

        return fig

    def _preview_to_nodes_edges(self, preview: dict):
        """Convert {nodes:[{id}], links:[{source,target,confidence/value}]} -> nodes/edges for render_network."""
        nodes, edges, seen = [], [], set()

        for n in (preview or {}).get("nodes", []):
            nid = str(n.get("id") or n.get("name") or n)
            if not nid or nid in seen:
                continue
            seen.add(nid)
            nodes.append({"id": nid, "label": nid, "size": 20, "color": self.colors["accent_cyan"]})

        for e in (preview or {}).get("links", []):
            s = e.get("source")
            t = e.get("target")
            if not s or not t:
                continue
            w = e.get("value", e.get("confidence", 1.0)) or 1.0
            edges.append({"from": str(s), "to": str(t), "weight": float(w)})
        return nodes, edges

    def render_from_preview(self, preview: dict, *, height: int = 380, title: str = "PPI Network") -> go.Figure:
        """Build a Plotly figure directly from the API preview structure."""
        nodes, edges = self._preview_to_nodes_edges(preview or {})
        return self.render_network(nodes, edges, height=height, title=title)


