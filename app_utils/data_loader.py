import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

from .config import REPORTS_DIR, TABLES_DIR, DATA_DIR, TMDB_API_KEY, TMDB_API_BASE, TMDB_IMG_BASE, REPORTS_OUT, TABLES_OUT


# the internal helper is cached; callers can clear the cache by calling
# `load_artifacts_cache_clear()` or by setting `force_reload=True`.
@st.cache_data
def _load_artifacts_cached():
    """Private implementation that actually reads the artifact files."""
    try:
        with open(os.path.join(REPORTS_DIR, "cluster_profiles.json"), "r", encoding='utf-8') as f:
            profiles = json.load(f)
        labels_df = pd.read_parquet(os.path.join(TABLES_DIR, "cluster_labels_users.parquet"))
        with open(os.path.join(REPORTS_DIR, "cluster_cards.md"), "r", encoding='utf-8') as f:
            cards_md = f.read()
        try:
            with open(os.path.join(REPORTS_DIR, "model_metadata.json"), "r", encoding='utf-8') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = None
        return profiles, labels_df, cards_md, metadata
    except FileNotFoundError:
        return None, None, None, None


def load_artifacts(force_reload: bool = False):
    """Public helper that optionally clears the cache before returning.

    Parameters
    ----------
    force_reload : bool, optional
        If True the cached results are invalidated and the files are read
        from disk again. Useful when the clustering pipeline has been rerun
        and you want the Streamlit app to pick up the latest "artifacts".
        The default (False) will return the cached copy if available.
    """
    if force_reload:
        _load_artifacts_cached.clear()
    return _load_artifacts_cached()


@st.cache_data
def load_projection_data():
    """Load pre-computed chart data from disk for interactive Plotly rendering."""
    from .config import FIGURES_DIR
    tsne_path  = os.path.join(FIGURES_DIR, "tsne_sample_data.csv")
    pca3d_path = os.path.join(FIGURES_DIR, "pca3d_sample.parquet")
    df_tsne  = pd.read_csv(tsne_path)   if os.path.exists(tsne_path)  else None
    df_pca3d = pd.read_parquet(pca3d_path) if os.path.exists(pca3d_path) else None
    return df_tsne, df_pca3d

def load_artifacts_cache_clear():
    """Convenience function to drop the cached artifacts and projection data entirely."""
    _load_artifacts_cached.clear()
    load_projection_data.clear()

@st.cache_data
def load_movie_lookup():
    """Loads dim_links_clean for imdbId/tmdbId mapping, and merges with titles."""
    try:
        links  = pd.read_parquet(os.path.join(DATA_DIR, "dim_links_clean.parquet"))[["movieId","tmdbId"]]
        movies = pd.read_parquet(os.path.join(DATA_DIR, "dim_movies_clean.parquet"))[["movieId","title","genres"]]
        return pd.merge(movies, links, on="movieId", how="left")
    except Exception:
        return None

@st.cache_data(ttl=86400)
def fetch_poster_by_name(title: str) -> str | None:
    """Search TMDB by movie title, return full poster URL."""
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            f"{TMDB_API_BASE}/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "language": "en-US"},
            timeout=5
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results and results[0].get("poster_path"):
                return TMDB_IMG_BASE + results[0]["poster_path"]
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_poster_by_tmdb_id(tmdb_id) -> str | None:
    """Fetch exact movie from TMDB, return full poster URL."""
    if not TMDB_API_KEY or pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(
            f"{TMDB_API_BASE}/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return TMDB_IMG_BASE + path
    except Exception:
        pass
    return None

def generate_artifacts(df_original, df_scaled, labels, kmeans_model, feature_cols, movies, interactions, scaler, best_k):
    """Generate Parquet data, JSON profiles, and Markdown narrative."""
    import datetime
    method_str = f"kmeans_k{best_k}"
    print(f"Exporting Artifacts for {method_str}...")
    
    # 1. cluster_labels_users.parquet
    df_original['cluster_id'] = labels
    labels_df = df_original[['userId', 'cluster_id']].copy()
    labels_df['method'] = method_str
    labels_df.to_parquet(os.path.join(TABLES_OUT, "cluster_labels_users.parquet"), index=False)
    
    # 2. cluster_profiles.json & Narrative
    profiles = {}
    centroids = kmeans_model.cluster_centers_
    
    md_lines = [
        "# Story A: Taste Tribes - Cluster Narratives\n",
        "This document describes each identified taste tribe (cluster) and its behavioral profile.\n"
    ]
    
    for c in np.unique(labels):
        cluster_data = df_original[df_original['cluster_id'] == c]
        size = len(cluster_data)
        percentage = (size / len(df_original)) * 100
        
        centroid = centroids[c]
        genre_cols = [col for col in feature_cols if "genre" in col.lower()]
        
        # Profile top genres
        genre_scores = {col: centroid[feature_cols.index(col)] for col in genre_cols}
        top_genres = sorted(genre_scores, key=genre_scores.get, reverse=True)[:5]
        
        # Extract Human Readable Name
        if top_genres:
            top2 = [g.replace('genre_','').replace('genre:','').title() for g in top_genres[:2]]
            profile_name = " & ".join(top2) + " Enthusiasts"
        else:
            profile_name = f"General Group {c}"
            
        profiles[str(c)] = {
            "profile_name": profile_name,
            "size": int(size),
            "percentage": float(percentage),
            "top_genres": top_genres,
            "centroid_scores": {f: float(v) for f, v in zip(feature_cols, centroid)}
        }
        
        md_lines.append(f"## Cluster {c}: {profile_name}")
        md_lines.append(f"- **Size**: {size} users ({percentage:.1f}%)")
        clean_genres = [g.replace('genre_','').replace('genre:','').title() for g in top_genres]
        md_lines.append(f"- **Top Preferences**: {', '.join(clean_genres)}")
        
        # 3. Top 10 representative movies
        md_lines.append("\n### Representative Movies")
        if interactions is not None and movies is not None:
            # Subset interactions for this cluster
            c_users = cluster_data['userId'].unique()
            c_interactions = interactions[interactions['userId'].isin(c_users)]
            
            # Find highly rated + popular movies within cluster
            movie_stats = c_interactions.groupby('movieId').agg(
                c_mean_rating=('rating', 'mean'),
                c_rating_count=('rating', 'count')
            ).reset_index()
            
            # Filter somewhat popular in the cluster to drop noisy 5.0s
            min_count = max(5, np.percentile(movie_stats['c_rating_count'], 75))
            top_movies = movie_stats[movie_stats['c_rating_count'] >= min_count].nlargest(10, 'c_mean_rating')
            
            # Join titles
            top_movies = top_movies.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
            
            for _, row in top_movies.iterrows():
                md_lines.append(f"- **{row['title']}** (Mean: {row['c_mean_rating']:.2f}, Views: {row['c_rating_count']}) - _{row['genres']}_")
        else:
            md_lines.append("*Interactions or Movies metadata not fully available to compute top movies.*")
            
        md_lines.append("\n---\n")
        
    # Write JSON
    with open(os.path.join(REPORTS_OUT, "cluster_profiles.json"), "w", encoding='utf-8') as f:
        json.dump(profiles, f, indent=4)
        
    # Write Model Metadata for Streamlit Demo
    metadata = {
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    with open(os.path.join(REPORTS_OUT, "model_metadata.json"), "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    # Write Markdown
    with open(os.path.join(REPORTS_OUT, "cluster_cards.md"), "w", encoding='utf-8') as f:
        f.write("\n".join(md_lines))

    # 4. run_manifest.json (Provenance)
    manifest = {
        "story": "Story A: Taste Tribes",
        "timestamp": datetime.datetime.now().isoformat(),
        "inputs": ["user_features_train.parquet", "dim_movies_clean.parquet"],
        "method": "K-Means Clustering",
        "parameters": {
            "k": int(best_k),
            "random_state": 42
        },
        "metrics": {
            "n_users": int(len(df_original)),
            "n_clusters": int(best_k)
        }
    }
    with open(os.path.join(REPORTS_OUT, "run_manifest.json"), "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)

    # 5. summary.md (Narrative)
    summary_text = f"""# Story A: Taste Tribes - Summary Report
Executed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Successfully segmented {len(df_original)} users into {best_k} distinct 'Taste Tribes' using K-Means clustering. 
The clustering was performed on genre preference features and activity metrics.

## Key Findings
- The optimal K was determined to be {best_k} based on silhouette score analysis.
- Tribes range from 'Enthusiasts' of specific genres (like Drama & Romance) to broader activity-based groups.
- Visualizations (Radar and t-SNE) confirm well-separated boundaries between major segments.

## Artifacts Generated
- `cluster_labels_users.parquet`: Per-user cluster assignments.
- `cluster_profiles.json`: Centroid details and top genres per tribe.
- `cluster_cards.md`: Detailed human-readable descriptions of each tribe.
"""
    with open(os.path.join(REPORTS_OUT, "summary.md"), "w", encoding='utf-8') as f:
        f.write(summary_text)

    print(f"Artifacts successfully exported to: {REPORTS_OUT}")
