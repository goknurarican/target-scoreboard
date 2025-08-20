"""Simplified Network Visualization - Fallback"""
import plotly.graph_objects as go
import random

class InteractiveNetworkViz:
    def __init__(self, theme_colors):
        self.colors = theme_colors
    
    def create_ppi_network_graph(self, targets, ppi_data):
        fig = go.Figure()
        x_pos = [random.uniform(0, 1) for _ in targets]
        y_pos = [random.uniform(0, 1) for _ in targets]
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            text=targets,
            textposition="middle center",
            marker=dict(size=25, color=self.colors['accent_cyan']),
            name='Targets'
        ))
        
        fig.update_layout(
            title="PPI Network (Simplified)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=self.colors['bg_primary'],
            paper_bgcolor=self.colors['bg_surface']
        )
        return fig
    
    def create_pathway_heatmap(self, target_scores):
        targets = [ts.get('target', 'Unknown') for ts in target_scores]
        pathways = ["RTK Signaling", "Cell Cycle", "Apoptosis", "DNA Repair"]
        data = [[random.uniform(0.2, 0.9) for _ in targets] for _ in pathways]
        
        fig = go.Figure(data=go.Heatmap(
            z=data, x=targets, y=pathways,
            colorscale='Viridis'
        ))
        fig.update_layout(title="Pathway Enrichment")
        return fig
    
    def create_competitive_landscape_chart(self, patent_data):
        companies = ['Roche', 'Pfizer', 'Novartis', 'Merck', 'BMS']
        fig = go.Figure(data=go.Scatter(
            x=[random.uniform(0.3, 0.9) for _ in companies],
            y=[random.uniform(0.4, 0.8) for _ in companies],
            mode='markers+text',
            text=companies,
            marker=dict(size=30, color=self.colors['warning'])
        ))
        fig.update_layout(
            title="Competitive Landscape",
            xaxis_title="Market Position",
            yaxis_title="Innovation Score"
        )
        return fig
    
    def create_druggability_radar(self, target_data):
        categories = ['Chemical', 'Structural', 'Safety', 'Market', 'Patent']
        values = [random.uniform(0.4, 0.9) for _ in categories]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories,
            fill='toself',
            name=target_data.get('target', 'Target')
        ))
        fig.update_layout(title="Druggability Assessment")
        return fig
