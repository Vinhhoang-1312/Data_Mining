"""
MovieLens Data Mining — Unified Dashboard
Covers:
  • Story A: Taste Tribes (K-Means Clustering)
  • Story C: Behavioral Weirdness (Anomaly Detection)

Run with: streamlit run app.py
"""

import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="MovieLens Data Mining",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Import Story A custom CSS (covers .section-header, genre cards, movie cards)
from app_utils.config import TMDB_API_KEY
import app_utils.ui_components as ui
ui.inject_custom_css()

# Extra styles for Story C metric cards
st.markdown("""
<style>
.metric-card {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 MovieLens Mining")
    st.markdown("---")
    page = st.radio(
        "Chọn Story",
        options=["Story A: Taste Tribes", "Story C: Behavioral Weirdness"],
        index=0,
    )
    st.markdown("---")


# ── Helper Functions ──────────────────────────────────────────────────────────
def _genre_display_name(profile):
    """Extract a human-readable tribe name from genre features only."""
    if "profile_name" in profile:
        return profile["profile_name"].replace("Pref__", "").replace("genre_", "").replace("genre_pref__", "").title()
    top = profile.get("top_genres", [])
    genre_only = [g for g in top if 'genre_pref__' in g]
    if not genre_only:
        genre_only = [g for g in top if 'genre' in g.lower()]
    if genre_only:
        names = [g.replace('genre_pref__', '').replace('genre_', '').replace('_', ' ').title() for g in genre_only[:2]]
        return ' & '.join(names) + ' Lovers'
    return f"Tribe {profile.get('cluster_id', '?')}"

def _genre_only_prefs(profile):
    """Return only genre-related items from top_genres/top_preferences."""
    top = profile.get("top_preferences", profile.get("top_genres", []))
    genre_only = [g for g in top if 'genre_pref__' in g or 'genre' in g.lower()]
    return [g.replace('genre_pref__', '').replace('genre_', '').replace('_', ' ').title() for g in genre_only]

def _safe_metric(val):
    """Safely format a metric value with comma separator."""
    if isinstance(val, (int, float)):
        return f"{int(val):,}"
    return str(val)


# ══════════════════════════════════════════════════════════════════════════════
# STORY A — TASTE TRIBES
# ══════════════════════════════════════════════════════════════════════════════
if page == "Story A: Taste Tribes":
    from app_utils.config import TABLES_DIR, FIGURES_DIR
    from app_utils.data_loader import load_artifacts, load_movie_lookup, load_projection_data
    from app_utils.logic import (
        build_synthetic_user_vector,
        find_nearest_cluster,
        parse_movies_from_markdown,
        project_user_into_charts,
    )
    from app_utils.visualizations import build_tsne_fig_with_user, build_pca3d_fig_with_user

    # Sidebar controls for Story A
    with st.sidebar:
        if st.button("🔄 Reload artifacts"):
            from app_utils.data_loader import load_artifacts_cache_clear
            load_artifacts_cache_clear()
            st.rerun()

    profiles, labels_df, cards_md, metadata = load_artifacts()
    movie_lookup = load_movie_lookup()

    if not profiles:
        st.warning("Artifacts not found! Run notebook `story_a_taste_tribes.ipynb` first.")
        st.stop()

    df_tsne_base, df_pca3d_base = load_projection_data()

    # Sidebar info for Story A
    with st.sidebar:
        st.write(f"- **Method:** K-Means")
        st.write(f"- **Optimal K:** {len(profiles)}")
        if TMDB_API_KEY:
            st.success("✅ TMDB API connected")
        else:
            st.warning("⚠️ Set TMDB_API_KEY in .env")
        st.markdown("---")
        labels_path = os.path.join(TABLES_DIR, "cluster_labels_users.parquet")
        if os.path.exists(labels_path):
            st.download_button(
                "📥 Download Cluster Labels",
                data=open(labels_path, "rb").read(),
                file_name="cluster_labels_users.parquet",
                mime="application/octet-stream",
            )

    # Title
    st.title("🎬 Story A: Taste Tribes")
    st.markdown("User segmentation — phân nhóm người dùng thành các nhóm sở thích.")

    active_tab = st.radio("View", ["📊 Overview & Analytics", "🚀 Cold-Start Demo"], horizontal=True, label_visibility="collapsed")

    # ── Tab 1: Overview ──────────────────────────────────────────────────────
    if active_tab == "📊 Overview & Analytics":
        st.markdown('<div class="section-header">Tribe Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            ui.render_pie_chart(profiles)
        with col2:
            with st.expander("📄 Cluster Cards", expanded=True):
                display_cards_md = cards_md.replace("Pref__", "").replace("genre_", "")
                st.markdown(display_cards_md)

        st.markdown('<div class="section-header">Genre Fingerprints (Radar)</div>', unsafe_allow_html=True)
        radar_path = os.path.join(FIGURES_DIR, "genre_radar.html")
        if os.path.exists(radar_path):
            with open(radar_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500)
        else:
            st.info("Radar chart not found — re-run the notebook.")

        st.markdown('<div class="section-header">2D Manifold Projection (t-SNE)</div>', unsafe_allow_html=True)
        if df_tsne_base is not None:
            st.plotly_chart(build_tsne_fig_with_user(df_tsne_base), use_container_width=True, key="a_overview_tsne")
        else:
            tsne_path = os.path.join(FIGURES_DIR, "cluster_scatter_tsne.html")
            if os.path.exists(tsne_path):
                with open(tsne_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=600)
            else:
                st.info("t-SNE scatter not found — re-run the notebook.")

        st.markdown('<div class="section-header">3D Cluster Projection (PCA)</div>', unsafe_allow_html=True)
        if df_pca3d_base is not None:
            st.plotly_chart(build_pca3d_fig_with_user(df_pca3d_base), use_container_width=True, key="a_overview_pca3d")
        else:
            pca_path = os.path.join(FIGURES_DIR, "cluster_scatter_pca_3d.html")
            if os.path.exists(pca_path):
                with open(pca_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=650)
            else:
                st.info("3D scatter not found — re-run the notebook.")

    # ── Tab 2: Cold-Start Demo ───────────────────────────────────────────────
    elif active_tab == "🚀 Cold-Start Demo":
        st.markdown("## 🎯 Cold-Start User Simulation")
        st.markdown("Chọn thể loại phim yêu thích để xem bạn thuộc nhóm nào.")
        st.markdown("---")

        if not metadata:
            st.warning("Model metadata not found — re-run the notebook.")
            st.stop()

        if "selected_genres" not in st.session_state:
            st.session_state.selected_genres = set()
        if "user_tsne"   not in st.session_state:
            st.session_state.user_tsne   = None
        if "user_pca3d"  not in st.session_state:
            st.session_state.user_pca3d  = None

        def toggle_genre(g):
            if g in st.session_state.selected_genres:
                st.session_state.selected_genres.discard(g)
            else:
                if len(st.session_state.selected_genres) < 5:
                    st.session_state.selected_genres.add(g)

        feature_cols = metadata["feature_cols"]
        genre_cols   = [c for c in feature_cols if "genre_pref__" in c]
        display_genres = [c.replace("genre_pref__", "").replace("_", "-").title() for c in genre_cols]

        ui.render_genre_selector_grid(display_genres, toggle_genre)

        n_selected = len(st.session_state.selected_genres)
        if n_selected > 0:
            pills = "  ".join([f"`{g}`" for g in st.session_state.selected_genres])
            st.markdown(f"**Selected ({n_selected}/5):** {pills}")
        else:
            st.caption("Chưa chọn thể loại nào — chọn ít nhất 1.")

        st.markdown("---")
        go = st.button("🔍 Find My Taste Tribe!", type="primary", disabled=(n_selected == 0))

        if go and n_selected > 0:
            with st.spinner("Đang tính toán..."):
                user_scaled = build_synthetic_user_vector(
                    st.session_state.selected_genres,
                    feature_cols,
                    metadata["scaler_mean"],
                    metadata["scaler_scale"],
                )
                best_cluster, chosen_profile = find_nearest_cluster(user_scaled, profiles, feature_cols)
                movie_items = parse_movies_from_markdown(cards_md, best_cluster)
                user_tsne, user_pca3d = project_user_into_charts(user_scaled, FIGURES_DIR)
                st.session_state.user_tsne      = user_tsne
                st.session_state.user_pca3d     = user_pca3d
                st.session_state.best_cluster   = best_cluster
                st.session_state.chosen_profile = chosen_profile
                st.session_state.movie_items    = movie_items

        # Persist results across reruns (genre select triggers rerun)
        if "best_cluster" in st.session_state and st.session_state.best_cluster is not None:
            best_cluster   = st.session_state.best_cluster
            chosen_profile = st.session_state.chosen_profile
            movie_items    = st.session_state.get("movie_items", [])

            st.markdown("---")
            display_name = _genre_display_name(chosen_profile)
            st.markdown(f'<div class="tribe-badge">Cluster {best_cluster}</div>', unsafe_allow_html=True)
            st.markdown(f"## 🎉 You belong to: **{display_name}**")
            genre_prefs = _genre_only_prefs(chosen_profile)
            if genre_prefs:
                st.caption("Top genre preferences: " + ", ".join(genre_prefs))

            st.markdown('<div class="section-header">🎬 Your Recommended Movies</div>', unsafe_allow_html=True)
            if movie_items:
                ui.render_recommended_movies(movie_items, movie_lookup)
            elif movie_lookup is not None and not movie_lookup.empty:
                # Fallback: recommend top-rated movies matching the cluster's top genres
                top_genre_names = [g.replace('genre_pref__', '').replace('_', '-').title()
                                   for g in chosen_profile.get('top_genres', [])
                                   if 'genre_pref__' in g]
                if top_genre_names:
                    mask = movie_lookup['genres'].str.contains('|'.join(top_genre_names), case=False, na=False)
                    recs = movie_lookup[mask].drop_duplicates('title').head(10)
                    if not recs.empty:
                        fallback_items = [
                            {"title": row['title'], "info": row.get('genres', ''), "genre": '', "raw": ''}
                            for _, row in recs.iterrows()
                        ]
                        ui.render_recommended_movies(fallback_items, movie_lookup)
                    else:
                        st.info("Không tìm thấy phim phù hợp.")
                else:
                    st.info("Không tìm thấy phim phù hợp.")
            else:
                st.info("Movie lookup data not available — re-run the notebook.")

        st.markdown("---")
        st.markdown("### 📍 Where do you sit among the Tribes?")
        st.caption("Submit your genres above to see your position (⭐) in the charts.")

        if df_tsne_base is not None or df_pca3d_base is not None:
            col_2d, col_3d = st.columns(2)
            with col_2d:
                st.markdown("**2D t-SNE Projection**")
                if df_tsne_base is not None:
                    st.plotly_chart(
                        build_tsne_fig_with_user(df_tsne_base, user_tsne=st.session_state.get("user_tsne")),
                        use_container_width=True, key="a_coldstart_tsne",
                    )
            with col_3d:
                st.markdown("**3D PCA Projection**")
                if df_pca3d_base is not None:
                    st.plotly_chart(
                        build_pca3d_fig_with_user(df_pca3d_base, user_pca3d=st.session_state.get("user_pca3d")),
                        use_container_width=True, key="a_coldstart_pca3d",
                    )


# ══════════════════════════════════════════════════════════════════════════════
# STORY C — BEHAVIORAL WEIRDNESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Story C: Behavioral Weirdness":
    STORY_C_DIR = os.path.join("artifacts", "story_C")
    TABLES_DIR  = os.path.join(STORY_C_DIR, "tables")
    REPORTS_DIR = os.path.join(STORY_C_DIR, "reports")
    FIGURES_DIR = os.path.join(STORY_C_DIR, "figures")

    @st.cache_data
    def load_story_c_artifacts():
        user_path     = os.path.join(TABLES_DIR, "user_anomaly_scores.parquet")
        movie_path    = os.path.join(TABLES_DIR, "movie_anomaly_scores.parquet")
        manifest_path = os.path.join(REPORTS_DIR, "run_manifest.json")
        summary_path  = os.path.join(REPORTS_DIR, "summary.md")
        cases_path    = os.path.join(REPORTS_DIR, "case_studies.md")
        return (
            pd.read_parquet(user_path)  if os.path.exists(user_path)     else None,
            pd.read_parquet(movie_path) if os.path.exists(movie_path)    else None,
            json.load(open(manifest_path, encoding='utf-8')) if os.path.exists(manifest_path) else {},
            open(summary_path, encoding='utf-8').read()   if os.path.exists(summary_path)  else "",
            open(cases_path, encoding='utf-8').read()     if os.path.exists(cases_path)    else "",
        )

    user_scores, movie_scores, manifest, summary_md, cases_md = load_story_c_artifacts()

    # Sidebar info for Story C
    with st.sidebar:
        if manifest:
            st.caption("Last run:")
            st.write(manifest.get("timestamp", "?")[:19])
            metrics = manifest.get("metrics", {})
            st.metric("Users analysed",  _safe_metric(metrics.get('n_users_sampled', '?')))
            st.metric("Anomalies (ISO)", _safe_metric(metrics.get('n_anomalous_iso', '?')))
            st.metric("Movies analysed", _safe_metric(metrics.get('n_movies_analyzed', '?')))
        else:
            st.warning("Run notebook `story_c_behavioral_weirdness.ipynb` first.")

    # Title
    st.title("🔍 Story C: Behavioral Weirdness")
    st.markdown("Xác định **người dùng bất thường** (Isolation Forest + LOF) và **phim gây phân cực** (Robust Std Ranking).")

    if user_scores is None or movie_scores is None:
        st.error("Artifacts not found. Run notebook `story_c_behavioral_weirdness.ipynb` first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["👤 User Anomalies", "🎬 Polarizing Movies", "📄 Reports"])

    # ── Tab 1: User Anomalies ────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">User Anomaly Scatter</div>', unsafe_allow_html=True)
        scatter_path = os.path.join(FIGURES_DIR, "user_anomaly_scatter.html")
        if os.path.exists(scatter_path):
            with open(scatter_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=520)
        else:
            st.info("Scatter chart not found — re-run the notebook.")

        st.markdown('<div class="section-header">Top Anomalous Users</div>', unsafe_allow_html=True)
        n_top = st.slider("Show top N users", 5, 50, 10)
        cols_show = ["rank", "userId", "combined_score", "iso_forest_score", "lof_score", "iso_forest_label"]
        cols_avail = [c for c in cols_show if c in user_scores.columns]
        df_display = user_scores.head(n_top)[cols_avail].copy()
        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
        st.dataframe(df_display, use_container_width=True)

        user_parquet = os.path.join(TABLES_DIR, "user_anomaly_scores.parquet")
        if os.path.exists(user_parquet):
            st.download_button(
                "📥 Download User Anomaly Scores",
                data=open(user_parquet, "rb").read(),
                file_name="user_anomaly_scores.parquet",
                mime="application/octet-stream",
            )

    # ── Tab 2: Polarizing Movies ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Polarization Score Histogram</div>', unsafe_allow_html=True)
        hist_path = os.path.join(FIGURES_DIR, "movie_polarization_hist.html")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=480)
        else:
            st.info("Histogram not found — re-run the notebook.")

        st.markdown('<div class="section-header">Top Polarizing Movies</div>', unsafe_allow_html=True)
        n_movies = st.slider("Show top N movies", 10, 100, 20)
        movie_cols = ["rank", "title", "genres", "rating_mean", "rating_std", "n_ratings", "polarization_score"]
        movie_cols_avail = [c for c in movie_cols if c in movie_scores.columns]
        df_movies_display = movie_scores.head(n_movies)[movie_cols_avail].copy()
        for col in ["rating_mean", "rating_std", "polarization_score"]:
            if col in df_movies_display.columns:
                df_movies_display[col] = df_movies_display[col].round(3)
        st.dataframe(df_movies_display, use_container_width=True)

        movie_parquet = os.path.join(TABLES_DIR, "movie_anomaly_scores.parquet")
        if os.path.exists(movie_parquet):
            st.download_button(
                "📥 Download Movie Anomaly Scores",
                data=open(movie_parquet, "rb").read(),
                file_name="movie_anomaly_scores.parquet",
                mime="application/octet-stream",
            )

    # ── Tab 3: Reports ───────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Summary Report</div>', unsafe_allow_html=True)
        st.markdown(summary_md)
        st.markdown('<div class="section-header">Case Studies</div>', unsafe_allow_html=True)
        st.markdown(cases_md)
