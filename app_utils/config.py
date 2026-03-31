import os

# ── Directories ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts/story_A"
REPORTS_DIR   = os.path.join(ARTIFACTS_DIR, "reports")
TABLES_DIR    = os.path.join(ARTIFACTS_DIR, "tables")
FIGURES_DIR   = os.path.join(ARTIFACTS_DIR, "figures")
DATA_DIR      = "data-warehousing"

# For mining script compatibility (aliases)
TABLES_OUT = TABLES_DIR
REPORTS_OUT = REPORTS_DIR
FIGURES_OUT = FIGURES_DIR

# ── TMDB API ────────────────────────────────────────────────────────────────────
TMDB_API_KEY  = os.getenv("TMDB_API_KEY", "")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_BASE = "https://api.themoviedb.org/3"

# ── Static Lookup Data ──────────────────────────────────────────────────────────
GENRE_MOVIES = {
    "Action":      "The Dark Knight",
    "Adventure":   "Indiana Jones and the Raiders of the Lost Ark",
    "Animation":   "Spirited Away",
    "Children":    "Toy Story",
    "Comedy":      "The Grand Budapest Hotel",
    "Crime":       "Goodfellas",
    "Documentary": "Planet Earth II",
    "Drama":       "The Shawshank Redemption",
    "Fantasy":     "The Lord of the Rings: The Fellowship of the Ring",
    "Film-Noir":   "Chinatown",
    "Horror":      "Get Out",
    "Imax":        "Interstellar",
    "Musical":     "La La Land",
    "Mystery":     "Knives Out",
    "Romance":     "Pride & Prejudice",
    "Sci-Fi":      "The Matrix",
    "Thriller":    "Parasite",
    "War":         "Saving Private Ryan",
    "Western":     "Django Unchained",
}
