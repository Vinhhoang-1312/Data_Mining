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
import numpy as np
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
from app_utils.config import TMDB_API_KEY
import app_utils.ui_components as ui
ui.inject_custom_css()

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


# ── Helper Functions ──────────────────────────────────────────────────────────
def _genre_display_name(profile):
    # Prefer 'name' (Vietnamese or custom), then 'profile_name'
    name_val = profile.get("name") or profile.get("profile_name")
    if name_val:
        label = name_val.replace("Pref__", "").replace("genre_", "").replace("genre_pref__", "").title()
        return f"Tribe {profile.get('cluster_id', '?')}: {label}"
        
    top = profile.get("top_genres", [])
    genre_only = [g for g in top if 'genre_pref__' in g or 'genre' in g.lower() or len(g) < 20]
    if not genre_only and "centroid_scores" in profile:
        scores = profile["centroid_scores"]
        all_genres = {k: v for k, v in scores.items() if 'genre_pref__' in k or 'genre' in k.lower()}
        if all_genres:
            genre_only = sorted(all_genres, key=all_genres.get, reverse=True)[:2]
    names = [g.replace('genre_pref__', '').replace('genre_', '').replace('_', ' ').title() for g in genre_only[:2]]
    base_name = ' & '.join(names) or "Unknown"
    return f"Tribe {profile.get('cluster_id', '?')}: {base_name} Enthusiasts"

def _genre_only_prefs(profile):
    top = profile.get("top_preferences", profile.get("top_genres", []))
    genre_only = [g for g in top if 'genre_pref__' in g or 'genre' in g.lower()]
    return [g.replace('genre_pref__', '').replace('genre_', '').replace('_', ' ').title() for g in genre_only]

def _safe_metric(val):
    if isinstance(val, (int, float)):
        return f"{int(val):,}"
    return str(val)

# ── STORY A ───────────────────────────────────────────────────────────────────
def render_story_a():
    from app_utils.config import TABLES_DIR, FIGURES_DIR
    from app_utils.data_loader import (
        load_artifacts, load_movie_lookup, load_projection_data,
        load_user_features, load_cluster_labels, lookup_user_data,
    )
    from app_utils.logic import (
        build_synthetic_user_vector, find_nearest_cluster,
        parse_movies_from_markdown, project_user_into_charts,
    )
    from app_utils.visualizations import build_tsne_fig_with_user, build_pca3d_fig_with_user

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

    with st.sidebar:
        st.write(f"- **Method:** K-Means (Comparisons with GMM in Notebook)")
        st.write(f"- **Optimal K:** {len(profiles)}")
        if TMDB_API_KEY:
            st.success("✅ TMDB API connected")
        else:
            st.warning("⚠️ Set TMDB_API_KEY in .env")
        st.markdown("---")

    st.title("🎬 Story A: Taste Tribes")
    st.markdown("User segmentation — phân nhóm người dùng thành các nhóm sở thích.")

    active_tab = st.radio("View", ["📊 Overview & Analytics", "🚀 Cold-Start Demo", "🔍 User Lookup"], horizontal=True, label_visibility="collapsed")

    if active_tab == "📊 Overview & Analytics":
        st.markdown('<div class="section-header">Tribe Distribution</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.5])
        with col1: ui.render_pie_chart(profiles)
        with col2:
            with st.expander("📄 Cluster Cards", expanded=True):
                st.markdown(cards_md.replace("Pref__", "").replace("genre_", ""))

        st.markdown('<div class="section-header">2D Manifold Projection (t-SNE)</div>', unsafe_allow_html=True)
        if df_tsne_base is not None:
            st.plotly_chart(build_tsne_fig_with_user(df_tsne_base), use_container_width=True, key="a_overview_tsne", theme=None)

        st.markdown('<div class="section-header">3D Cluster Projection (PCA)</div>', unsafe_allow_html=True)
        if df_pca3d_base is not None:
            st.plotly_chart(build_pca3d_fig_with_user(df_pca3d_base), use_container_width=True, key="a_overview_pca3d", theme=None)

    elif active_tab == "🚀 Cold-Start Demo":
        render_cold_start_tab(metadata, profiles, cards_md, FIGURES_DIR, movie_lookup, df_tsne_base, df_pca3d_base)

    elif active_tab == "🔍 User Lookup":
        render_user_lookup_tab(profiles, metadata, movie_lookup, FIGURES_DIR, df_tsne_base, df_pca3d_base)


def render_cold_start_tab(metadata, profiles, cards_md, FIGURES_DIR, movie_lookup, df_tsne_base, df_pca3d_base):
    from app_utils.logic import build_synthetic_user_vector, find_nearest_cluster, parse_movies_from_markdown, project_user_into_charts
    from app_utils.visualizations import build_tsne_fig_with_user, build_pca3d_fig_with_user
    
    st.markdown("## 🎯 Cold-Start User Simulation")
    st.markdown("Chọn thể loại phim yêu thích để xem bạn thuộc nhóm nào.")
    st.markdown("---")

    if "selected_genres" not in st.session_state: st.session_state.selected_genres = set()
    
    def toggle_genre(g):
        initial_len = len(st.session_state.selected_genres)
        if g in st.session_state.selected_genres: st.session_state.selected_genres.discard(g)
        elif len(st.session_state.selected_genres) < 5: st.session_state.selected_genres.add(g)
        
        if len(st.session_state.selected_genres) != initial_len:
            st.session_state.best_cluster = None # Selection changed, clear results

    feature_cols = metadata["feature_cols"]
    genre_cols   = [c for c in feature_cols if "genre_pref__" in c]
    display_genres = [c.replace("genre_pref__", "").replace("_", "-").title() for c in genre_cols]

    # --- Presets & Reset ---
    c1, c2 = st.columns([3, 1])
    with c1:
        preset_label = st.selectbox("🚀 **Quick Presets:** Select a Tribe to see its typical genres", 
                                   ["-- Manual Selection --"] + [f"Tribe {k}: {(_genre_display_name(v)).split(':')[-1].strip()}" for k, v in profiles.items()],
                                   label_visibility="collapsed")
    with c2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.selected_genres = set()
            st.session_state.best_cluster = None
            st.rerun()

    if preset_label != "-- Manual Selection --":
        t_id = preset_label.split(":")[0].replace("Tribe ", "").strip()
        prof = profiles.get(t_id)
        if prof:
            top_g = prof.get("top_genres", [])
            new_sel = set([g.replace("genre_pref__", "").replace("genre_", "").replace("_", "-").title() for g in top_g[:5]])
            if new_sel != st.session_state.selected_genres:
                st.session_state.selected_genres = new_sel
                st.session_state.best_cluster = None # Clear old results
                st.rerun()

    st.markdown("---")

    # Fragment isolator for genre grid
    if hasattr(st, "fragment"):
        @st.fragment
        def isolated_genre_grid():
            ui.render_genre_selector_grid(display_genres, toggle_genre)
            n_sel = len(st.session_state.selected_genres)
            if n_sel > 0:
                pills = "  ".join([f"`{g}`" for g in st.session_state.selected_genres])
                st.markdown(f"**Selected ({n_sel}/5):** {pills}")
            else:
                st.caption("Chưa chọn thể loại nào.")
        isolated_genre_grid()
    else:
        ui.render_genre_selector_grid(display_genres, toggle_genre)

    st.markdown("---")
    if st.button("🔍 Find My Taste Tribe!", type="primary", disabled=(len(st.session_state.selected_genres) == 0)):
        with st.spinner("Đang tính toán..."):
            u_scaled = build_synthetic_user_vector(st.session_state.selected_genres, feature_cols, metadata["scaler_mean"], metadata["scaler_scale"])
            bc, cp = find_nearest_cluster(u_scaled, profiles, feature_cols, metadata["scaler_mean"], metadata["scaler_scale"])
            st.session_state.best_cluster = bc
            st.session_state.chosen_profile = cp
            st.session_state.movie_items = parse_movies_from_markdown(cards_md, bc)
            st.session_state.user_tsne, st.session_state.user_pca3d = project_user_into_charts(u_scaled, FIGURES_DIR, assigned_cluster=bc, profiles=profiles)

    if st.session_state.get("best_cluster") is not None:
        bc, cp = st.session_state.best_cluster, st.session_state.chosen_profile
        st.markdown(f'<div class="tribe-badge">Cluster {bc}</div>', unsafe_allow_html=True)
        st.markdown(f"## 🎉 You belong to: **{_genre_display_name(cp)}**")
        ui.render_recommended_movies(st.session_state.movie_items, movie_lookup)
        
        st.markdown("### 📍 Your Position")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(build_tsne_fig_with_user(df_tsne_base, user_tsne=st.session_state.user_tsne), use_container_width=True, key="cs_tsne")
        with c2: st.plotly_chart(build_pca3d_fig_with_user(df_pca3d_base, user_pca3d=st.session_state.user_pca3d), use_container_width=True, key="cs_pca3d")


def render_user_lookup_tab(profiles, metadata, movie_lookup, FIGURES_DIR, df_tsne_base, df_pca3d_base):
    from app_utils.data_loader import load_cluster_labels, load_user_features, lookup_user_data
    from app_utils.logic import project_user_into_charts
    from app_utils.visualizations import build_tsne_fig_with_user, build_pca3d_fig_with_user
    
    st.markdown("## 🔍 User Lookup")
    lbl_df = load_cluster_labels()
    feat_df = load_user_features()
    if lbl_df is None: return st.error("Missing data artifacts.")

    min_u, max_u = int(lbl_df.userId.min()), int(lbl_df.userId.max())
    uid = st.number_input("Enter userId", min_value=min_u, max_value=max_u, value=min_u)
    
    if st.button("🔍 Tra cứu") or st.session_state.get("lookup_uid") == uid:
        st.session_state.lookup_uid = uid
        cid, urow, rmov = lookup_user_data(uid, lbl_df, feat_df, movie_lookup)
        
        if cid is not None:
            chosen_profile = profiles.get(str(cid), profiles.get(cid, {}))
            st.markdown(f"## 🎉 User #{uid} belongs to: **{_genre_display_name(chosen_profile)}**")
            
            st.markdown("### 📊 Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Ratings", int(urow.get('n_ratings', 0)))
            m2.metric("Mean Rating", f"{float(urow.get('rating_mean', 0)):.2f}")
            m3.metric("Active Days", int(urow.get('active_days', 0)))

            st.markdown("### 🎬 Recently Rated")
            if rmov is not None:
                items = [{"title": r['title'], "info": r.get('genres',''), "genre":'','raw':''} for _, r in rmov.head(7).iterrows()]
                ui.render_recommended_movies(items, movie_lookup)
            else:
                st.caption("No rating history found.")

            # --- Projection Charts for User Lookup ---
            st.markdown("---")
            st.markdown(f"### 📍 User #{uid} Position in the Tribes")
            
            f_cols = metadata["feature_cols"]
            u_vec = np.array([urow.get(c, 0.0) for c in f_cols])
            u_scaled = (u_vec - np.array(metadata['scaler_mean'])) / np.array(metadata['scaler_scale'])
            ut, up = project_user_into_charts(u_scaled, FIGURES_DIR, assigned_cluster=cid, profiles=profiles)
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(build_tsne_fig_with_user(
                    df_tsne_base, user_tsne=ut, highlight_cluster=cid, user_label=f"User #{uid}"), 
                    use_container_width=True, key=f"lookup_tsne_{uid}")
            with c2:
                st.plotly_chart(build_pca3d_fig_with_user(
                    df_pca3d_base, user_pca3d=up, highlight_cluster=cid, user_label=f"User #{uid}"), 
                    use_container_width=True, key=f"lookup_pca3d_{uid}")


# ── STORY C ───────────────────────────────────────────────────────────────────
def render_story_c():
    STORY_C_DIR = os.path.join("artifacts", "story_C")
    TABLES_DIR  = os.path.join(STORY_C_DIR, "tables")
    REPORTS_DIR = os.path.join(STORY_C_DIR, "reports")
    FIGURES_DIR = os.path.join(STORY_C_DIR, "figures")
    
    @st.cache_data
    def load_story_c_data():
        upath = os.path.join(TABLES_DIR, "user_anomaly_scores.parquet")
        mpath = os.path.join(TABLES_DIR, "movie_anomaly_scores.parquet")
        json_path = os.path.join(REPORTS_DIR, "story_c_summary.json")
        sum_path = os.path.join(REPORTS_DIR, "story_c_case_studies.md")
        
        return (
            pd.read_parquet(upath) if os.path.exists(upath) else None,
            pd.read_parquet(mpath) if os.path.exists(mpath) else None,
            json.load(open(json_path, encoding='utf-8')) if os.path.exists(json_path) else {},
            open(sum_path, encoding='utf-8').read() if os.path.exists(sum_path) else ""
        )

    user_scores, movie_scores, manifest, summary_md = load_story_c_data()

    with st.sidebar:
        if manifest:
            st.metric("Total Users", _safe_metric(manifest.get('user_count', '?')))
            st.metric("Total Movies", _safe_metric(manifest.get('movie_count', '?')))
            if 'user_jaccard' in manifest:
                st.metric("Method Overlap", f"{manifest['user_jaccard']:.1%}")

    st.title("🎬 Story C: Behavioral Weirdness")
    st.markdown("Anomaly detection — tìm kiếm những người dùng có hành vi khác thường.")
    
    tab_user, tab_movie = st.tabs(["👤 User Anomalies", "📽️ Movie Polarities"])
    
    with tab_user:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("### 🗺️ Anomaly Landscape")
            scatter_path = os.path.join(FIGURES_DIR, "user_anomaly_scatter.html")
            if os.path.exists(scatter_path):
                with open(scatter_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=550, scrolling=True)
            else:
                st.info("Interactive scatter plot not found.")
        
        with c2:
            st.markdown("### 📊 Scoring Distributions")
            dist_path = os.path.join(FIGURES_DIR, "eval_user_score_distribution.png")
            if os.path.exists(dist_path):
                st.image(dist_path, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🔍 Methodology")
            st.info("""
            Outliers are detected using TWO independent methods:
            1. **Isolation Forest**: Multi-dimensional isolation score.
            2. **Robust Z-Score**: Extreme behavior in specific dimensions (MAD-based).
            """)

        st.markdown("---")
        st.markdown("### 🌌 3D Anomaly Projection (PCA)")
        pca_path = os.path.join(FIGURES_DIR, "user_pca_anomaly.html")
        if os.path.exists(pca_path):
            with open(pca_path, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=650)

        if user_scores is not None:
            st.markdown("---")
            st.markdown("### 📊 Top Anomalous Users Discovery")
            if 'if_label' in user_scores.columns:
                anomalies = user_scores[user_scores['if_label'] == -1].sort_values('if_score', ascending=False)
                st.dataframe(anomalies.head(50), use_container_width=True)
            else:
                st.dataframe(user_scores.head(20), use_container_width=True)

    with tab_movie:
        m1, m2 = st.columns([1, 1])
        with m1:
            st.markdown("### ⚖️ Mean Rating vs Std Dev")
            m_scatter = os.path.join(FIGURES_DIR, "movie_mean_vs_std.png")
            if os.path.exists(m_scatter):
                st.image(m_scatter, use_container_width=True)
        
        with m2:
            st.markdown("### � Polarization Index")
            m_hist = os.path.join(FIGURES_DIR, "movie_polarization_hist.html")
            if os.path.exists(m_hist):
                with open(m_hist, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=500)

        if movie_scores is not None:
            st.markdown("---")
            st.markdown("### 📽️ High Polarization & Unusual Movies")
            st.dataframe(movie_scores.head(50), use_container_width=True)

    if summary_md:
        with st.expander("📖 View detailed Behavioral Case Studies"):
            st.markdown(summary_md)

# ── MAIN ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 MovieLens Mining")
    page = st.radio("Chọn Story", ["Story A: Taste Tribes", "Story C: Behavioral Weirdness"])

if page == "Story A: Taste Tribes":
    render_story_a()
else:
    render_story_c()
