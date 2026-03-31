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


def find_nearest_cluster(user_scaled, profiles, feature_cols):
    """
    Computes Euclidean distance between the simulated user vector and all KMeans centroids.
    Returns the nearest cluster ID (best_cluster) and distance.
    """
    best_cluster, min_dist = None, float('inf')
    
    for k, v in profiles.items():
        centroid = np.array([v["centroid_scores"][col] for col in feature_cols])
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
        if s.startswith(f"## Cluster {cluster_id}:"):
            in_cluster = True
        elif in_cluster and s.startswith("## Cluster"):
            break
        elif in_cluster and s.startswith("- **") and "(Mean:" in s:
            try:
                title_raw = s.split("**")[1]
                genre_tag = s.split("_")[-1].strip() if "_" in s else ""
                info      = s.split("(Mean:")[1].split(")")[0].strip() if "(Mean:" in s else ""
                movie_items.append({
                    "title": title_raw, 
                    "info": info, 
                    "genre": genre_tag, 
                    "raw": s
                })
            except Exception:
                pass
                
    return movie_items


def project_user_into_charts(user_scaled, figures_dir):
    """
    Projects a cold-start user's scaled feature vector into the pre-computed
    2D t-SNE and 3D PCA chart spaces.

    Strategy:
    - 3D PCA: uses the saved PCA model to transform exactly.
    - 2D t-SNE: t-SNE is non-parametric, so we approximate by finding the
      nearest neighbor in the saved t-SNE sample and using its coordinates
      with a small offset so the user marker is visually distinct.

    Parameters
    ----------
    user_scaled : np.ndarray
        The scaled feature vector produced by `build_synthetic_user_vector`.
    figures_dir : str
        Directory where projection artifacts live (pca_3d_model.joblib, tsne_sample_data.csv).

    Returns
    -------
    user_tsne : tuple (x, y) or None
    user_pca3d : tuple (x, y, z) or None
    """
    import pandas as pd

    user_tsne = None
    user_pca3d = None

    # ── 3D PCA (exact projection) ──
    pca_model_path = os.path.join(figures_dir, "pca_3d_model.joblib")
    pca3d_sample_path = os.path.join(figures_dir, "pca3d_sample.parquet")
    if os.path.exists(pca_model_path) and os.path.exists(pca3d_sample_path):
        try:
            import joblib
            pca_3d = joblib.load(pca_model_path)
            # pca_3d was fitted on X_sample; user_scaled may have more features
            # We need to align: the PCA n_features_ tells us how many it expects
            n_feat = pca_3d.n_features_in_
            arr = np.array(user_scaled[:n_feat]).reshape(1, -1)
            coords_3d = pca_3d.transform(arr)[0]
            user_pca3d = tuple(coords_3d)
        except Exception:
            user_pca3d = None

    # ── 2D t-SNE (nearest-neighbor approximation) ──
    tsne_path = os.path.join(figures_dir, "tsne_sample_data.csv")
    if os.path.exists(tsne_path):
        try:
            df_tsne = pd.read_csv(tsne_path)
            # We only stored x, y, Cluster — no raw features, so we use the
            # cluster assignment as a guide: find the centroid of the user's
            # assigned cluster in t-SNE space and add a tiny offset.
            # (If we stored raw features in tsne_sample we could do KNN;
            #  for now cluster-centroid is a reliable, fast approximation.)
            # Derive cluster from pca3d sample if available
            if user_pca3d is not None and os.path.exists(pca3d_sample_path):
                df_pca3d = pd.read_parquet(pca3d_sample_path)
                # Find the Cluster label of the nearest 3D neighbor
                diffs = (df_pca3d[['x', 'y', 'z']].values
                         - np.array(user_pca3d))
                dists = np.linalg.norm(diffs, axis=1)
                nearest_cluster = df_pca3d.iloc[dists.argmin()]['Cluster']
                # Get mean t-SNE coords for that cluster
                cluster_tsne = df_tsne[df_tsne['Cluster'].astype(str) == str(nearest_cluster)]
                if not cluster_tsne.empty:
                    cx = float(cluster_tsne['x'].mean())
                    cy = float(cluster_tsne['y'].mean())
                    # Small offset so star doesn't sit exactly on top of the cluster center
                    user_tsne = (cx + 3.0, cy + 3.0)
        except Exception:
            user_tsne = None

    return user_tsne, user_pca3d
