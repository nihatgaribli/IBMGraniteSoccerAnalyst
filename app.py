"""
Soccer Tactical Analysis - Streamlit Application

An interactive web application for visualizing and analyzing soccer tracking data
using Voronoi diagrams and AI-powered tactical explanations.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import warnings

# Import our custom modules from src package
from src.soccer_tracking_data import get_soccer_tracking_data, get_world_cup_matches
from src.spatial_analysis import SoccerVoronoiAnalyzer
from src.tactical_explainer import TacticalExplainer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Soccer Tactical Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .explanation-box {
        background-color: #fff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_world_cup_matches():
    """
    Load and cache World Cup 2022 matches.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping match names to match IDs
    """
    return get_world_cup_matches()


@st.cache_data
def load_tracking_data(match_id: int):
    """
    Load and cache tracking data from StatsBomb.
    
    Parameters
    ----------
    match_id : int
        StatsBomb match ID
    
    Returns
    -------
    pd.DataFrame
        Tracking data for all shots in the match
    """
    try:
        with st.spinner('Loading tracking data from StatsBomb...'):
            df = get_soccer_tracking_data(match_id=match_id)
        return df
    except Exception as e:
        st.error(f"Error loading tracking data: {e}")
        return pd.DataFrame()


def create_soccer_pitch():
    """
    Create a Plotly figure with a soccer pitch background.
    
    Returns
    -------
    go.Figure
        Plotly figure with pitch markings
    """
    fig = go.Figure()
    
    # Pitch dimensions
    pitch_length = 120
    pitch_width = 80
    
    # Pitch outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=pitch_length, y1=pitch_width,
        line=dict(color="white", width=2),
        fillcolor="rgba(34, 139, 34, 0.3)"  # Green with transparency
    )
    
    # Center line
    fig.add_shape(
        type="line",
        x0=pitch_length/2, y0=0,
        x1=pitch_length/2, y1=pitch_width,
        line=dict(color="white", width=2)
    )
    
    # Center circle
    fig.add_shape(
        type="circle",
        x0=pitch_length/2 - 10, y0=pitch_width/2 - 10,
        x1=pitch_length/2 + 10, y1=pitch_width/2 + 10,
        line=dict(color="white", width=2)
    )
    
    # Penalty areas
    # Left penalty area
    fig.add_shape(
        type="rect",
        x0=0, y0=pitch_width/2 - 22,
        x1=18, y1=pitch_width/2 + 22,
        line=dict(color="white", width=2)
    )
    
    # Right penalty area
    fig.add_shape(
        type="rect",
        x0=pitch_length - 18, y0=pitch_width/2 - 22,
        x1=pitch_length, y1=pitch_width/2 + 22,
        line=dict(color="white", width=2)
    )
    
    # Goal areas
    # Left goal area
    fig.add_shape(
        type="rect",
        x0=0, y0=pitch_width/2 - 10,
        x1=6, y1=pitch_width/2 + 10,
        line=dict(color="white", width=2)
    )
    
    # Right goal area
    fig.add_shape(
        type="rect",
        x0=pitch_length - 6, y0=pitch_width/2 - 10,
        x1=pitch_length, y1=pitch_width/2 + 10,
        line=dict(color="white", width=2)
    )
    
    # Attacking third line (X = 80)
    fig.add_shape(
        type="line",
        x0=80, y0=0,
        x1=80, y1=pitch_width,
        line=dict(color="yellow", width=2, dash="dash")
    )
    
    # Configure layout with fixed dimensions and proper aspect ratio
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='rgba(34, 139, 34, 0.8)',  # Green background
        paper_bgcolor='white',
        width=800,
        height=533,
        xaxis=dict(
            range=[0, pitch_length],  # Fixed range 0-120
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            title="Length (yards)",
            constrain='domain'
        ),
        yaxis=dict(
            range=[0, pitch_width],  # Fixed range 0-80
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            title="Width (yards)",
            scaleanchor="x",
            scaleratio=1,
            constrain='domain'
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=False
    )
    
    # Ensure proper aspect ratio (3:2 for football pitch)
    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(
        constrain='domain',
        scaleanchor="x",
        scaleratio=1
    )
    
    return fig


def plot_voronoi_on_pitch(shot_df, analyzer):
    """
    Plot Voronoi diagram on soccer pitch using Plotly.
    
    Parameters
    ----------
    shot_df : pd.DataFrame
        DataFrame for a single shot event
    analyzer : SoccerVoronoiAnalyzer
        Initialized analyzer instance
    
    Returns
    -------
    go.Figure
        Plotly figure with Voronoi cells and players
    """
    # Create base pitch
    fig = create_soccer_pitch()
    
    # Get player positions
    points = shot_df[['x', 'y']].values
    
    if len(points) < 3:
        st.warning("Not enough players for Voronoi diagram (minimum 3 required)")
        return fig
    
    # Add boundary points
    extended_points = analyzer._add_boundary_points(points)
    
    # Compute Voronoi
    try:
        vor = Voronoi(extended_points)
    except Exception as e:
        st.error(f"Error computing Voronoi: {e}")
        return fig
    
    # Get unique teams in the current shot
    unique_teams = shot_df['team_name'].unique()
    
    # Dynamically determine attacking and defending teams
    # Attacking team is typically the first team or the one with 'Teammate' designation
    attacking_team = None
    defending_team = None
    
    for team in unique_teams:
        if team in ['Teammate', 'Argentina']:
            attacking_team = team
        else:
            defending_team = team
    
    # If no explicit match, use first team as attacking, second as defending
    if attacking_team is None and len(unique_teams) > 0:
        attacking_team = unique_teams[0]
        defending_team = unique_teams[1] if len(unique_teams) > 1 else unique_teams[0]
    elif defending_team is None and len(unique_teams) > 1:
        defending_team = unique_teams[1]
    
    # Create dynamic team color mapping
    team_colors = {}
    team_markers = {}
    
    for team in unique_teams:
        if team == attacking_team:
            team_colors[team] = 'rgba(135, 206, 250, 0.4)'  # Light blue for Voronoi
            team_markers[team] = {'color': 'blue', 'symbol': 'circle'}
        else:
            team_colors[team] = 'rgba(255, 99, 71, 0.4)'  # Red for Voronoi
            team_markers[team] = {'color': 'red', 'symbol': 'square'}
    
    # Plot Voronoi cells for each player
    for idx, row in shot_df.iterrows():
        point_idx = shot_df.index.get_loc(idx)
        team_name = row['team_name']
        player_name = row['player_name']
        
        # Get Voronoi cell
        cell = analyzer._compute_voronoi_cell(vor, point_idx)
        
        if cell is not None and not cell.is_empty:
            # Get polygon coordinates
            x_coords, y_coords = cell.exterior.xy
            
            # Get color based on team
            color = team_colors.get(team_name, 'rgba(128, 128, 128, 0.4)')
            
            # Add polygon
            fig.add_trace(go.Scatter(
                x=list(x_coords),
                y=list(y_coords),
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                mode='lines',
                name=f"{player_name} ({team_name})",
                hoverinfo='name',
                showlegend=False
            ))
    
    # Plot player positions with proper legend grouping
    for team_name in unique_teams:
        team_df = shot_df[shot_df['team_name'] == team_name].copy()
        
        # Get marker properties from dynamic mapping
        marker_props = team_markers.get(team_name, {'color': 'gray', 'symbol': 'circle'})
        marker_color = marker_props['color']
        marker_symbol = marker_props['symbol']
        
        # Create shortened labels (first initial + last name or jersey number)
        short_labels = []
        for name in team_df['player_name']:
            parts = str(name).split()
            if len(parts) > 1:
                # Use first initial + last name (e.g., "L. Messi")
                short_label = f"{parts[0][0]}. {parts[-1]}"
            else:
                # Use first 8 characters if single name
                short_label = str(name)[:8]
            short_labels.append(short_label)
        
        # Add scatter plot for players
        fig.add_trace(go.Scatter(
            x=team_df['x'],
            y=team_df['y'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=marker_color,
                line=dict(color='white', width=2),
                symbol=marker_symbol
            ),
            text=short_labels,
            textposition='top center',
            textfont=dict(size=8, color='black', family='Arial Black'),
            name=team_name,
            customdata=team_df['player_name'],
            hovertemplate='<b>%{customdata}</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))
    
    # Update title
    fig.update_layout(
        title=dict(
            text=f"Voronoi Spatial Analysis - Shot {shot_df['shot_id'].iloc[0][:8]}...",
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig


def main():
    """
    Main Streamlit application.
    """
    # Header
    st.markdown('<div class="main-header">⚽ Soccer Tactical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">FIFA World Cup 2022 - Voronoi Spatial Analysis with AI-Powered Insights</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Configuration")
    st.sidebar.markdown("---")
    
    # Load World Cup matches
    st.sidebar.markdown("### 🏆 Match Selection")
    with st.spinner('Loading World Cup 2022 matches...'):
        world_cup_matches = load_world_cup_matches()
    
    # Match dropdown
    if not world_cup_matches:
        st.error("Unable to load World Cup matches. Please check your connection.")
        return
    
    # Create sorted list of match names
    match_names = sorted(world_cup_matches.keys())
    
    # Find default match (Argentina vs France final)
    default_match = next((m for m in match_names if "Argentina vs France" in m), match_names[0])
    default_index = match_names.index(default_match)
    
    selected_match_name = st.sidebar.selectbox(
        "Select a Match:",
        match_names,
        index=default_index,
        help="Choose a FIFA World Cup 2022 match to analyze"
    )
    
    # Get the match ID for the selected match
    match_id = world_cup_matches[selected_match_name]
    
    # Display match info
    st.sidebar.info(f"**Match ID:** {match_id}")
    
    # Load tracking data for selected match
    tracking_df = load_tracking_data(match_id)
    
    if tracking_df.empty:
        st.error("No tracking data available. Please check the match ID.")
        return
    
    # Shot selection
    st.sidebar.markdown("### 📊 Shot Selection")
    
    # Get unique shots with timestamps
    shot_info = tracking_df.groupby('shot_id').agg({
        'timestamp': 'first'
    }).reset_index()
    
    # Create display labels
    shot_labels = [
        f"Shot {i+1}: {row['timestamp']} ({row['shot_id'][:8]}...)"
        for i, row in shot_info.iterrows()
    ]
    
    selected_shot_label = st.sidebar.selectbox(
        "Select a shot event:",
        shot_labels,
        help="Choose a shot to analyze"
    )
    
    # Get selected shot index
    selected_idx = shot_labels.index(selected_shot_label)
    selected_shot_id = shot_info.iloc[selected_idx]['shot_id']
    
    # Filter data for selected shot
    shot_df = tracking_df[tracking_df['shot_id'] == selected_shot_id].copy()
    
    # Display shot info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Shot Details")
    st.sidebar.write(f"**Shot ID:** {selected_shot_id[:16]}...")
    st.sidebar.write(f"**Timestamp:** {shot_df['timestamp'].iloc[0]}")
    st.sidebar.write(f"**Players Tracked:** {len(shot_df)}")
    
    # Team breakdown
    team_counts = shot_df['team_name'].value_counts()
    st.sidebar.markdown("**Team Breakdown:**")
    for team, count in team_counts.items():
        st.sidebar.write(f"  • {team}: {count} players")
    
    # Initialize analyzer (no caching - compute fresh each time)
    analyzer = SoccerVoronoiAnalyzer(
        pitch_length=120.0,
        pitch_width=80.0,
        attacking_third_start=80.0
    )
    
    # Perform spatial analysis (recompute for each shot selection)
    with st.spinner('Computing Voronoi spatial analysis...'):
        analysis_result = analyzer.analyze_shot_event(shot_df)
    
    # Initialize tactical explainer (no caching)
    explainer = TacticalExplainer(use_mock=True)
    
    # Generate tactical explanation (recompute for each shot selection)
    # This ensures the explanation updates when dropdown changes
    with st.spinner('Generating AI tactical explanation...'):
        explanation = explainer.generate_explanation(analysis_result)
    
    # Main content area - Two columns
    col1, col2 = st.columns([2, 1])
    
    # Column 1: Voronoi visualization
    with col1:
        st.markdown("### 🗺️ Spatial Dominance Map")
        
        # Create and display Voronoi plot
        fig = plot_voronoi_on_pitch(shot_df, analyzer)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🔵 **Blue circles:** Attacking Team (Possession / Active Phase)
        - 🔴 **Red squares:** Defending Team (Structure / Block)
        - 🟡 **Yellow dashed line:** Attacking Third Boundary ($X = 80$)
        - 🟩 **Colored polygons:** Voronoi Cells (Instantaneous Spatial Control Area)
        """)
    
    # Column 2: Metrics and explanation
    with col2:
        st.markdown("### 📊 Tactical Dashboard")
        
        # Team dominance metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">ATTACKING THIRD DOMINANCE</div>', unsafe_allow_html=True)
        
        attacking_dom = analysis_result['team_dominance']['Attacking Team']
        defending_dom = analysis_result['team_dominance']['Defending Team']
        
        # Create two sub-columns for metrics
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="Attacking Team",
                value=f"{attacking_dom:.1f}%",
                delta=f"{attacking_dom - 50:.1f}%" if attacking_dom > 50 else None
            )
        
        with metric_col2:
            st.metric(
                label="Defending Team",
                value=f"{defending_dom:.1f}%",
                delta=f"{defending_dom - 50:.1f}%" if defending_dom > 50 else None
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pitch coverage
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Pitch Coverage",
            value=f"{analysis_result['total_pitch_coverage']:.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Player areas
        st.markdown("### 👥 Player Controlled Areas")
        
        # Sort players by area
        player_areas = analysis_result['player_areas']
        sorted_players = sorted(player_areas.items(), key=lambda x: x[1], reverse=True)
        
        # Display top 5 players
        st.markdown("**Top 5 Players:**")
        for i, (player, area) in enumerate(sorted_players[:5], 1):
            st.write(f"{i}. {player}: {area:.0f} sq yards")
        
        # AI Tactical Explanation
        st.markdown("### 🤖 AI Tactical Analysis")
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.write(explanation)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional insights
        with st.expander("📈 Detailed Metrics"):
            st.json({
                "Shot ID": selected_shot_id,
                "Attacking Third Coverage": {
                    "Attacking Team Area": f"{analysis_result['attacking_third_coverage']['Attacking Team Area']:.2f} sq yards",
                    "Defending Team Area": f"{analysis_result['attacking_third_coverage']['Defending Team Area']:.2f} sq yards",
                    "Total Controlled": f"{analysis_result['attacking_third_coverage']['Total Controlled']:.2f} sq yards"
                },
                "Player Count": len(shot_df),
                "Timestamp": shot_df['timestamp'].iloc[0]
            })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Built with using Streamlit, Plotly, and IBM Granite AI<br>
        Data source: StatsBomb Open Data
        By Nihat Garibli
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Made with Bob
