import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .config import FIGURES_OUT
import joblib

def apply_dimensionality_reduction(X_scaled, labels):
    """Apply PCA and t-SNE for 2D visualization."""
    print("Applying Dimensionality Reduction (PCA & t-SNE)...")
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    print(f"PCA explained variance ratio (2 comps): {pca.explained_variance_ratio_.sum():.4f}")
    
    # 2. t-SNE for plotting (sample to 1,000 for performance)
    sample_size = min(1000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled.iloc[idx]
    labels_sample = labels[idx]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X_sample)
    
    df_tsne = pd.DataFrame({'x': tsne_result[:, 0], 'y': tsne_result[:, 1], 'Cluster': labels_sample})
    df_tsne['Cluster'] = df_tsne['Cluster'].astype(str)
    
    # Save t-SNE sample data for cold-start user projection
    df_tsne.to_csv(os.path.join(FIGURES_OUT, "tsne_sample_data.csv"), index=False)
    
    fig = px.scatter(
        df_tsne, x='x', y='y', color='Cluster',
        # title='Taste Tribes (t-SNE 2D Projection)', # Removed title to avoid overlap with legend
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # styling tweaks to avoid overlapping legend/text
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, sans-serif', size=12, color='#FFF'),
        legend=dict(
            orientation='h', 
            yanchor='bottom', y=1.05, 
            xanchor='center', x=0.5,
            font=dict(family='Inter, sans-serif', size=11, color='#FFF')
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    fig.write_html(os.path.join(FIGURES_OUT, "cluster_scatter_tsne.html"))
    # fig.write_image(os.path.join(FIGURES_OUT, "cluster_map.png")) # Disabled: hangs in this environment

    # 3. PCA 3D for plotting
    pca_3d = PCA(n_components=3)
    pca_3d_result = pca_3d.fit_transform(X_sample)
    
    # Save PCA 3D model for cold-start user projection
    joblib.dump(pca_3d, os.path.join(FIGURES_OUT, "pca_3d_model.joblib"))
    
    df_pca_3d = pd.DataFrame({'x': pca_3d_result[:, 0], 'y': pca_3d_result[:, 1], 'z': pca_3d_result[:, 2], 'Cluster': labels_sample})
    df_pca_3d['Cluster'] = df_pca_3d['Cluster'].astype(str)
    
    fig_3d = px.scatter_3d(
        df_pca_3d, x='x', y='y', z='z', color='Cluster' if 'Cluster' in df_pca_3d.columns else 'cluster',
        title='Taste Tribes (PCA 3D Projection)',
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, sans-serif', size=12, color='white'),
        title_font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=0.9, xanchor='right', x=1, font=dict(color='white')),
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="#444", showbackground=True, zerolinecolor="#666",
                title=dict(font=dict(color='white')), tickfont=dict(color='white')
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="#444", showbackground=True, zerolinecolor="#666",
                title=dict(font=dict(color='white')), tickfont=dict(color='white')
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="#444", showbackground=True, zerolinecolor="#666",
                title=dict(font=dict(color='white')), tickfont=dict(color='white')
            )
        )
    )
    fig_3d.write_html(os.path.join(FIGURES_OUT, "cluster_scatter_pca_3d.html"))

    # Save PCA 3D chart data so the Streamlit app can rebuild the chart interactively
    df_pca_3d.to_parquet(os.path.join(FIGURES_OUT, "pca3d_sample.parquet"), index=False)
    print("Projection artifacts saved (pca_3d_model.joblib, tsne_sample_data.csv, pca3d_sample.parquet)")


def build_tsne_fig_with_user(df_tsne: pd.DataFrame, user_tsne: tuple = None,
                               highlight_cluster=None, user_label="You") -> go.Figure:
    """Return a Plotly t-SNE scatter figure, optionally with a highlighted cold-start user point
    or a highlighted cluster ring (for user-lookup mode).

    Parameters
    ----------
    df_tsne : pd.DataFrame
        Must have columns 'x', 'y', 'Cluster' (loaded from tsne_sample_data.csv).
    user_tsne : tuple (x, y) or None
        If provided, adds a gold star marker at those coordinates.
    highlight_cluster : int or str or None
        If provided, adds a bright overlay trace that marks all points belonging to
        that cluster with a larger, semi-transparent ring — useful in User Lookup tab.
    user_label : str
        Text label for the user marker.
    """
    candidates = ['Cluster', 'cluster', 'Taste Tribe', 'Tribe']
    color_col = next((c for c in candidates if c in df_tsne.columns), 'Cluster')
    df_tsne[color_col] = df_tsne[color_col].astype(str)
    
    fig = px.scatter(
        df_tsne, x='x', y='y', color=color_col,
        # title='Taste Tribes (t-SNE 2D Projection)', # Removed title to avoid overlap with legend
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, sans-serif', size=12, color='#FFF'),
        legend=dict(
            orientation='h', 
            yanchor='bottom', y=1.05, 
            xanchor='center', x=0.5,
            font=dict(family='Inter, sans-serif', size=11, color='#FFF')
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    # Highlight a specific cluster (User Lookup mode)
    if highlight_cluster is not None:
        mask = df_tsne[color_col] == str(highlight_cluster)
        df_hi = df_tsne[mask]
        if not df_hi.empty:
            fig.add_trace(go.Scatter(
                x=df_hi['x'], y=df_hi['y'],
                mode='markers',
                marker=dict(symbol='circle-open', size=14, color='#FFD700',
                            line=dict(color='#FFD700', width=2)),
                name=f'Tribe {highlight_cluster} (User)', showlegend=True
            ))
    # User star point (Lookup or Cold-start)
    if user_tsne is not None:
        fig.add_trace(go.Scatter(
            x=[user_tsne[0]], y=[user_tsne[1]],
            mode='markers+text',
            marker=dict(symbol='star', size=24, color='#FFD700',
                        line=dict(color='#FF4500', width=3)),
            text=[user_label], textposition='top center',
            textfont=dict(color='#FFD700', size=13),
            name='You',
            showlegend=True
        ))
    return fig


def build_pca3d_fig_with_user(df_pca3d: pd.DataFrame, user_pca3d: tuple = None,
                                highlight_cluster=None, user_label="You") -> go.Figure:
    """Return a Plotly PCA 3D scatter figure, optionally with a highlighted cold-start user point
    or a highlighted cluster ring (for user-lookup mode).

    Parameters
    ----------
    df_pca3d : pd.DataFrame
        Must have columns 'x', 'y', 'z', 'Cluster' (loaded from pca3d_sample.parquet).
    user_pca3d : tuple (x, y, z) or None
        If provided, adds a gold diamond marker at those coordinates.
    highlight_cluster : int or str or None
        If provided, adds a bright overlay trace for all points of that cluster.
    user_label : str
        Text label for the user marker.
    """
    candidates = ['Cluster', 'cluster', 'Taste Tribe', 'Tribe']
    color_col = next((c for c in candidates if c in df_pca3d.columns), 'Cluster')
    df_pca3d[color_col] = df_pca3d[color_col].astype(str)

    fig_3d = px.scatter_3d(
        df_pca3d, x='x', y='y', z='z', color=color_col,
        title='Taste Tribes (PCA 3D Projection)',
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, sans-serif', size=12, color='white'),
        title_font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=0.9, xanchor='right', x=1, font=dict(color='white')),
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="#444", showbackground=True, zerolinecolor="#666",
                title=dict(font=dict(color='white')), tickfont=dict(color='white')
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="#444", showbackground=True, zerolinecolor="#666",
                title=dict(font=dict(color='white')), tickfont=dict(color='white')
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)", gridcolor="#444", showbackground=True, zerolinecolor="#666",
                title=dict(font=dict(color='white')), tickfont=dict(color='white')
            )
        )
    )
    # Highlight a specific cluster (User Lookup mode)
    if highlight_cluster is not None:
        mask = df_pca3d[color_col] == str(highlight_cluster)
        df_hi = df_pca3d[mask]
        if not df_hi.empty:
            fig_3d.add_trace(go.Scatter3d(
                x=df_hi['x'], y=df_hi['y'], z=df_hi['z'],
                mode='markers',
                marker=dict(symbol='circle-open', size=6, color='#FFD700',
                            line=dict(color='#FFD700', width=2)),
                name=f'Tribe {highlight_cluster} (User)', showlegend=True
            ))
    # User diamond point (Lookup or Cold-start)
    if user_pca3d is not None:
        fig_3d.add_trace(go.Scatter3d(
            x=[user_pca3d[0]], y=[user_pca3d[1]], z=[user_pca3d[2]],
            mode='markers+text',
            marker=dict(symbol='diamond', size=12, color='#FFD700',
                        line=dict(color='#FF4500', width=3)),
            text=[user_label], textposition='top center',
            textfont=dict(color='#FFD700', size=13),
            name='You',
            showlegend=True
        ))
    return fig_3d


def plot_silhouette_scores(k_range, scores):
    """Save silhouette score plot as PNG."""
    plt.figure(figsize=(8,5))
    plt.plot(k_range, scores, marker='o')
    plt.title('Silhouette Score vs. K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_OUT, "silhouette_scores.png"))
    plt.close()

def create_radar_chart(centroids, feature_cols):
    """Create radar chart comparing genre preferences across clusters."""
    print("Generating Radar Chart...")
    
    # Extract only genre-related features
    genre_cols = [c for c in feature_cols if "genre" in c.lower()]
    if not genre_cols:
        genre_cols = feature_cols[:8] # fallback
        
    top_n = min(8, len(genre_cols))
    selected_genres = genre_cols[:top_n]
    
    fig = go.Figure()
    
    for i, centroid in enumerate(centroids):
        # Retrieve the centroid scores for the selected genres
        values = [centroid[feature_cols.index(c)] for c in selected_genres]
        values.append(values[0]) # close polygon
        
        # Clean up labels heavily to avoid overlap
        labels = [g.lower().replace('genre_', '').replace('genre:', '').replace('pref__', '').replace('_', ' ').strip().title() for g in selected_genres]
        labels.append(labels[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=f'Tribe {i}',
            line=dict(width=3),
            opacity=0.7
        ))
        
    fig.update_layout(
      template="plotly_dark",
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      colorway=px.colors.qualitative.Vivid,
      polar=dict(
          bgcolor="rgba(30, 30, 40, 0.5)",
          radialaxis=dict(
              visible=True, 
              title="Preference Score", 
              tickfont=dict(size=11, color='#AAA'), 
              gridcolor="#444",
              linecolor="#444",
              angle=45
          ),
          angularaxis=dict(
              tickfont=dict(size=14, color='#FFF', family='Inter, sans-serif'), 
              gridcolor="#555", 
              linecolor="#555"
          )
      ),
      font=dict(family='Inter, sans-serif', size=13, color='#FFF'),
      showlegend=True,
      legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
      title=dict(
          text="Tribe Genre Fingerprints (Relative Preference)", 
          font=dict(size=18, color='#FFD700'),
          x=0.5,
          xanchor='center'
      ),
      margin=dict(l=80, r=80, t=60, b=80),
    )
    fig.write_html(os.path.join(FIGURES_OUT, "genre_radar.html"))
    # fig.write_image(os.path.join(FIGURES_OUT, "genre_radar.png")) # Disabled: hangs in this environment
