import streamlit as st
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import GENRE_MOVIES
from .data_loader import fetch_poster_by_name, fetch_poster_by_tmdb_id

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FAFAFA; }

    /* Title */
    h1 { color: #FFFFFF !important; letter-spacing: -0.5px; }

    /* Genre card grid */
    .genre-card {
        position: relative; border-radius: 8px; overflow: hidden;
        cursor: pointer; width: 100%; aspect-ratio: 2/3;
        border: 2px solid #333;
        transition: transform 0.2s ease, border-color 0.2s ease;
        background: #1a1a2a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .genre-card:hover { transform: scale(1.04); border-color: #3498db; z-index: 10; box-shadow: 0 6px 16px rgba(0,0,0,0.5); }
    .genre-card.selected { border-color: #3498db !important; transform: scale(1.04); background: #2a2a3a; }
    
    .genre-card img { 
        width: 100%; height: 100%; object-fit: cover; 
        display: block; position: absolute; top:0; left:0; 
        z-index: 1;
    }
    
    .genre-card .label {
        position: absolute; bottom: 0; left: 0; right: 0;
        background: rgba(0, 0, 0, 0.75);
        color: #fff; font-size: 14px; font-weight: 600;
        padding: 8px; text-align: center;
        backdrop-filter: blur(4px);
        z-index: 3;
    }
    .genre-card .check {
        position: absolute; top: 8px; right: 8px;
        background: #3498db; border-radius: 50%;
        width: 24px; height: 24px; display: flex;
        align-items: center; justify-content: center;
        font-size: 14px; color: white; font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
        z-index: 4;
    }

    /* Column spacing */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        margin-bottom: 24px;
    }

    /* Movie poster grid */
    .movie-card {
        position: relative; border-radius: 6px; overflow: hidden;
        width: 100%; aspect-ratio: 2/3;
        background: #1a1a2a;
        transition: transform 0.2s;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid #333;
    }
    .movie-card:hover { transform: scale(1.04); z-index: 10; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
    .movie-card img { width: 100%; height: 100%; object-fit: cover; display: block; position: absolute; top:0; left:0; z-index: 1; }

    /* Tribe badge */
    .tribe-badge {
        display: inline-block; background: linear-gradient(90deg, #3498db, #2980b9);
        color: white; font-size: 14px; font-weight: 600;
        padding: 6px 16px; border-radius: 20px; margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
    }

    .section-header {
        font-size: 20px; font-weight: 600; color: #FFFFFF;
        border-bottom: 2px solid #333333; padding-bottom: 8px;
        margin: 24px 0 16px;
    }

    /* Skeleton Animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 0.9; }
        100% { opacity: 0.6; }
    }
    .skeleton {
        background: #2a2a3a;
        animation: pulse 1.5s infinite;
        border-radius: 6px;
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
        z-index: 2;
    }
    </style>
    """, unsafe_allow_html=True)

def render_pie_chart(profiles):
    """Renders the cluster distribution pie chart."""
    def _profile_display_name(v):
        # Prefer 'name' if provided by user/notebook, then 'profile_name'
        name_val = v.get("name") or v.get("profile_name")
        if name_val:
            return f"Tribe {v.get('cluster_id', '?')}: " + name_val.replace('Pref__', '').replace('genre_', '').replace('genre_pref__', '').title()
            
        top = v.get("top_genres", [])
        # Handle both raw 'animation' and 'genre_pref__animation'
        genre_only = [g for g in top if 'genre_pref__' in g or 'genre' in g.lower() or len(g) < 20]
        if not genre_only and "centroid_scores" in v:
            scores = v["centroid_scores"]
            all_genres = {k: val for k, val in scores.items() if 'genre_pref__' in k or 'genre' in k.lower()}
            if all_genres:
                genre_only = sorted(all_genres, key=all_genres.get, reverse=True)[:2]
        names = [g.replace('genre_pref__', '').replace('genre_', '').replace('_', ' ').title() for g in genre_only[:2]]
        base_name = ' & '.join(names) or "Unknown"
        return f"Tribe {v.get('cluster_id', '?')}: {base_name} Enthusiasts"

    dist_data = [{"Name": _profile_display_name(v), "Size": v.get("size", v.get("n_users", 0))} for v in profiles.values()]
    fig_pie = px.pie(
        pd.DataFrame(dist_data), values='Size', names='Name', hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(
        textposition='outside',
        textinfo='percent+label',
        textfont=dict(size=11, family='Inter, sans-serif', color='#FFF')
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, sans-serif', color="#FFF"), 
        legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=20, r=20, t=30, b=20),
        autosize=True
    )
    st.plotly_chart(fig_pie, width='stretch')

def render_genre_selector_grid(display_genres, toggle_func):
    """Renders the clickable genre grid with ZERO-JITTER for cached posters."""
    if 'poster_cache' not in st.session_state:
        st.session_state.poster_cache = {}

    genre_list = [g for g in display_genres if g is not None]
    COLS = 7
    genre_rows = [genre_list[i:i+COLS] for i in range(0, len(genre_list), COLS)]
    
    genres_to_fetch = []
    
    for row in genre_rows:
        cols = st.columns(COLS)
        for idx, g in enumerate(row):
            with cols[idx]:
                is_selected = g in st.session_state.selected_genres
                cached_url = st.session_state.poster_cache.get(g)
                
                sel_class = "selected" if is_selected else ""
                chk_html = '<div class="check">✓</div>' if is_selected else ""
                
                if cached_url:
                    st.markdown(f"""<div class="genre-card {sel_class}">
                        <img src="{cached_url}" alt="{g}">
                        {chk_html}
                        <div class="label">{g}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="genre-card">
                        <div class="skeleton"></div>
                        <div style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:32px;z-index:3;">🎥</div>
                        <div class="label">{g}</div>
                        </div>""", unsafe_allow_html=True)
                    genres_to_fetch.append(g)
                
                lab = "✓ Selected" if is_selected else "+ Select"
                if st.button(lab, key=f"btn_{g}", width='stretch', type="primary" if is_selected else "secondary"):
                    toggle_func(g)
                    st.rerun()

    # Pass 2: Background updates for missing ones (will show on next interaction)
    def _fetch_one(g_name):
        url = fetch_poster_by_name(GENRE_MOVIES.get(g_name, g_name + " movie"))
        return g_name, url

    if genres_to_fetch:
        with ThreadPoolExecutor(max_workers=5) as executor:
            for g_name, url in executor.map(_fetch_one, genres_to_fetch):
                if url: st.session_state.poster_cache[g_name] = url

def render_recommended_movies(movie_items, movie_lookup):
    """Renders recommendations with instant-render for cached posters."""
    if not movie_items:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; border: 1px dashed rgba(255,255,255,0.2); text-align: center;">
            <div style="font-size: 40px; margin-bottom: 10px;">📽️</div>
            <div style="font-weight: 600; color: #CCC;">No specific representative movies found for this tribe yet.</div>
            <div style="font-size: 0.9em; color: #888; margin-top: 5px;">Try exploring other tribes or check back after re-running the notebook analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if 'poster_cache' not in st.session_state:
        st.session_state.poster_cache = {}

    MOVIE_COLS = 7
    movie_rows = [movie_items[i:i+MOVIE_COLS] for i in range(0, len(movie_items), MOVIE_COLS)]
    
    movie_slots = {}
    movies_to_fetch = []
    
    for row in movie_rows:
        cols = st.columns(MOVIE_COLS)
        for idx, movie in enumerate(row):
            m_title = movie['title']
            with cols[idx]:
                slot = st.empty()
                cached_url = st.session_state.poster_cache.get(m_title)
                if cached_url:
                    slot.markdown(f'<div class="movie-card"><img src="{cached_url}" alt="{m_title}"></div>', unsafe_allow_html=True)
                else:
                    slot.markdown(f'<div class="movie-card"><div class="skeleton"></div></div>', unsafe_allow_html=True)
                    movies_to_fetch.append(movie)
                st.markdown(f"**{m_title}**")
                movie_slots[m_title] = slot

    def _fetch_movie(m):
        m_title = m['title']
        search_name = m_title.split("(")[0].strip()[:20]
        matches = movie_lookup[movie_lookup["title"].str.contains(search_name, case=False, na=False, regex=False)]
        url = fetch_poster_by_tmdb_id(matches.iloc[0]["tmdbId"]) if not matches.empty else None
        return m_title, url

    if movies_to_fetch:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_fetch_movie, m): m for m in movies_to_fetch}
            for future in as_completed(futures):
                m_t, url = future.result()
                if url: st.session_state.poster_cache[m_t] = url
                slot = movie_slots.get(m_t)
                if slot:
                    img_html = f'<img src="{url}" alt="{m_t}">' if url else '<div class="skeleton"></div>'
                    slot.markdown(f'<div class="movie-card">{img_html}</div>', unsafe_allow_html=True)
