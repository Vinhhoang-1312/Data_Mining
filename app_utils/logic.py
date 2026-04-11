import numpy as np
import os

def build_synthetic_user_vector(selected_genres, feature_cols, scaler_mean, scaler_scale):
    """
    Simulates a "Cold Start" user by placing high score values (5.0) in the user's
    selected genres, and default values for standard metrics like n_ratings.
    """
    user_vector = np.zeros(len(feature_cols))
    
    # Typical baseline activity metrics
    if "n_ratings" in feature_cols:
        user_vector[feature_cols.index("n_ratings")] = np.log1p(20)
    if "rating_mean" in feature_cols:
        user_vector[feature_cols.index("rating_mean")] = 4.0

    # Max out the selected genres
    for g in selected_genres:
        raw_col = "genre_pref__" + g.lower().replace("-", "_")
        if raw_col in feature_cols:
            user_vector[feature_cols.index(raw_col)] = 5.0

    # Standardize it based on how the KMeans model was trained
    user_scaled = (user_vector - scaler_mean) / scaler_scale
    return user_scaled


def find_nearest_cluster(user_scaled, profiles, feature_cols, scaler_mean=None, scaler_scale=None):
    """
    Computes Euclidean distance between the simulated user vector and all KMeans centroids.
    Returns the nearest cluster ID (best_cluster) and distance.
    If actual centroids are missing, builds a proxy from 'top_genres' using scaler metadata.
    """
    best_cluster, min_dist = None, float('inf')
    
    for k, v in profiles.items():
        centroid_data = v.get("centroid_scores", v.get("centroid", {}))
        
        if not centroid_data:
            # Fallback: build proxy vector from top_genres
            top_g = [g.lower().replace("genre_pref__","").replace("genre_","").replace("-","_") for g in v.get("top_genres", [])]
            proxy_raw = np.zeros(len(feature_cols))
            for g in top_g:
                raw_c = "genre_pref__" + g
                if raw_c in feature_cols:
                    proxy_raw[feature_cols.index(raw_c)] = 5.0
            
            # Scale the proxy if metadata provided
            if scaler_mean is not None and scaler_scale is not None:
                centroid = (proxy_raw - scaler_mean) / scaler_scale
            else:
                centroid = proxy_raw
        else:
            centroid = np.array([centroid_data.get(col, 0.0) for col in feature_cols])
            
        d = np.linalg.norm(user_scaled - centroid)
        if d < min_dist:
            min_dist, best_cluster = d, k
            
    return best_cluster, profiles[best_cluster]


def parse_movies_from_markdown(cards_md, cluster_id):
    """
    Parses `cluster_cards.md` and strips out the list of representative 
    movies for the selected cluster_id.
    """
    lines = cards_md.split("\n")
    in_cluster, movie_items = False, []
    
    for line in lines:
        s = line.strip()
        # Be loose: look for "Cluster {id}" as a substring in a heading
        if s.startswith("## ") and f"Cluster {cluster_id}" in s:
            in_cluster = True
        elif in_cluster and s.startswith("## Cluster"):
            break
        elif in_cluster and s.startswith("- **"):
            try:
                title_raw = s.split("**")[1]
                info = ""
                if "(" in s: info = s.split("(")[1].split(")")[0].strip()
                elif "—" in s: info = s.split("—")[1].strip()
                
                movie_items.append({
                    "title": title_raw, 
                    "info": info, 
                    "raw": s
                })
            except Exception:
                pass
                
    return movie_items


def project_user_into_charts(user_scaled, figures_dir, assigned_cluster=None, profiles=None):
    """
    Projects a cold-start user's scaled feature vector into the pre-computed
    2D t-SNE and 3D PCA chart spaces.
    Robust matching: handles numeric IDs and descriptive Tribe names.
    """
    import pandas as pd
    import numpy as np

    user_tsne = None
    user_pca3d = None
    
    pca_model_path = os.path.join(figures_dir, "pca_3d_model.joblib")
    pca3d_sample_path = os.path.join(figures_dir, "pca3d_sample.parquet")
    tsne_path = os.path.join(figures_dir, "tsne_sample_data.csv")
    candidates = ['Cluster', 'cluster', 'Taste Tribe', 'Tribe']

    # ── 3D PCA ──
    if os.path.exists(pca_model_path):
        try:
            import joblib
            pca_3d = joblib.load(pca_model_path)
            n_feat = pca_3d.n_features_in_
            arr = np.array(user_scaled[:n_feat]).reshape(1, -1)
            user_pca3d = tuple(pca_3d.transform(arr)[0])
        except Exception: pass
        
    if user_pca3d is None and assigned_cluster is not None and os.path.exists(pca3d_sample_path):
        try:
            df_pca3d = pd.read_parquet(pca3d_sample_path)
            c_col = next((c for c in candidates if c in df_pca3d.columns), 'Cluster')
            mask = (df_pca3d[c_col].astype(str) == str(assigned_cluster))
            pts = df_pca3d[mask]
            if pts.empty and profiles is not None:
                # Name-based fallback
                name = profiles.get(str(assigned_cluster), {}).get("name", "")
                if name: pts = df_pca3d[df_pca3d[c_col].astype(str).str.contains(name, na=False)]
            
            if not pts.empty:
                user_pca3d = (float(pts['x'].mean()), float(pts['y'].mean()), float(pts['z'].mean()))
        except Exception: pass

    # ── 2D t-SNE ──
    if os.path.exists(tsne_path):
        try:
            df_tsne = pd.read_csv(tsne_path)
            t_col = next((c for c in candidates if c in df_tsne.columns), 'Cluster')
            mask = (df_tsne[t_col].astype(str) == str(assigned_cluster))
            pts = df_tsne[mask]
            if pts.empty and assigned_cluster is not None and profiles is not None:
                # Name-based fallback
                name = profiles.get(str(assigned_cluster), {}).get("name", "")
                if name: pts = df_tsne[df_tsne[t_col].astype(str).str.contains(name, na=False)]

            if not pts.empty:
                cx, cy = float(pts['x'].mean()), float(pts['y'].mean())
                user_tsne = (cx + 2.0, cy + 2.0)
        except Exception: pass

    return user_tsne, user_pca3d
