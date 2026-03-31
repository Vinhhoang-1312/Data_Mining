Group Final Project: MovieLens Data Mining - Project Proposal and Story Module Development Guide
🪶 By Nguyễn Sỹ Hùng, 2026
Note: This page is a practical landing page (not the official documentation). It can be treated as a proposal / starting point—each team member is free to adapt, extend, or develop their own story-module ideas as contributions. Moreover, we don't have to do all story modules A/B/C/D; we can focus on one or two and do them well. The key is to produce meaningful, interpretable insights from the MovieLens dataset using core data mining techniques.

Quick status overview:

Core pipeline & Preprocessed Data: Completed by Sỹ Hùng. All key mining-ready Parquet tables are built and available.
Story A module: Taste Tribes (clustering/segmentation) --> Assigned to Vĩnh Hoàng
Story B.1 module: Basket Liked-Movie Rules - Association (pattern mining) --> Assigned to Sỹ Hùng - in progress
Story B.2 module: Anti-Preference Rules (pattern mining) --> Assigned to Hưng Nguyễn - in progress
Story B.3 module: Temporal Shift Patterns (pattern mining) --> Not assigned yet
Story C module: Behavioral Weirdness (anomaly detection) --> not assigned yet
Story D module: Rating Foresight (predictive modeling) --> optional
Home
Intro
Pipeline
Tables
Story modules
Mining-Ready Data
Story Outputs
Project Proposal and Story Module Development Guide
Project Name: MovieLens Data Mining
Project Group Members: Sỹ Hùng, Vĩnh Hoàng, Hưng Nguyễn | Class: MSA30DN, FSB
Instructor: PhD. Cao Vu BUI
Introduction to the project
MovieLens Data Mining is a group final project focused on applying core data mining techniques to the MovieLens dataset (GroupLens / MovieLens.org), which contains timestamped user ratings and tags for movies.

The raw data comes as structured CSV files (easy to load), but like most real-world datasets it is sparse and imperfect, so it requires careful cleaning and leakage-safe preprocessing before mining.

The goal is to explore multiple aspects of the dataset and produce meaningful, interpretable insights that could support practical applications (e.g., recommendation, segmentation, explainable co-preference rules, and anomaly signals).

This project provides a completed, reproducible Core Pipeline that transforms MovieLens raw CSVs into Mining-Ready Parquet tables. Your job (as a story-module developer) is to consume those Parquet inputs, apply one family of mining methods, and produce interpretable outputs.

A story module is a downstream mining component (A/B/C/D) that consumes prebuilt Mining-Ready Parquet tables and produces both (1) machine-readable results (Parquet/JSON) and (2) a short narrative interpretation of findings.

Data Understanding:
Official MovieLens dataset: grouplens.org/datasets/movielens

Download the Parquet bundle here (Google Drive)

Key Mining-Ready inputs (quick view)
Parquet table	Built from	Purpose
interactions_train.parquet	fact_ratings_clean.parquet	Training-period ratings (time ≤ cutoff). Used to build features, baskets, and models without leakage.
interactions_test.parquet	fact_ratings_clean.parquet	Held-out later ratings (time > cutoff). Used for sanity checks and evaluation scaffolding.
user_features_train.parquet	interactions_train + dim_movies_clean (+ tags filtered by cutoff)	Per-user feature vectors for clustering and outlier detection (train-only).
movie_features_train.parquet	interactions_train + dim_movies_clean (+ tags filtered by cutoff)	Per-movie feature vectors for popularity, entropy, polarization, genre flags, and tag summaries (train-only).
transactions_train.parquet	interactions_train (likes) + dim_movies_clean (genre tokens) + optional tag tokens	Transaction database (basket/itemset) for frequent itemsets and association rules.
transactions_train_reduced.parquet	transactions_train.parquet (Phase 4.5 Reduction)	Mining-optimized baskets for Story B.1 (vocab-pruned + basket-pruned).
MovieLens Core Pipeline (backbone) overview
The core pipeline is already completed. This flow exists to provide context: the pipeline outputs are the inputs to story modules.

Project structure (for developers):
Part A — Core Pipeline (backbone): preprocess raw CSVs into clean, structured, Mining-Ready Parquet tables.
Part B — Story Modules (downstream mining applications): consume Mining-Ready Parquet inputs and produce mining outputs (clusters, rules, anomaly scores, predictions) plus short narrative interpretation.
Architecture Diagram
Architecture diagram showing the flow from raw CSVs to staging Parquet, clean contract tables, split, and feature engineering/transaction database, leading into the story modules (A/B/C/D).
Core Pipeline End-to-End Data Flow (Part A):
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                    MovieLens Core Pipeline (backbone): End-to-End Data Flow           │
│      Raw CSVs  →  Staging Parquet  →  Clean Contract Tables  →  Split  →              │
│      Feature Engineering & Transaction Database (basket/itemset representation)       │
└──────────────────────────────────────────────────────────────────────────────────────┘

PHASE 0 — RAW CSVs (untouched)
  data/raw/ml-latest/
    ratings.csv (userId, movieId, rating, timestamp)
    tags.csv    (userId, movieId, tag, timestamp)
    movies.csv  (movieId, title, genres)
    links.csv   (movieId, imdbId, tmdbId)
    (optional) genome-tags.csv (tagId, tag)
    (optional) genome-scores.csv (movieId, tagId, relevance)

PHASE 1 — STAGING PARQUET (Schema Lock + Provenance)
  data/raw_staging/
    ratings_raw.parquet        ← ratings.csv
    tags_raw.parquet           ← tags.csv
    movies_raw.parquet         ← movies.csv
    links_raw.parquet          ← links.csv
    genome_tags_raw.parquet    ← genome-tags.csv (optional)
    genome_scores_raw.parquet  ← genome-scores.csv (optional)

PHASE 2 — CLEAN CONTRACT TABLES (Normalize + Validate)
  tables/
    fact_ratings_clean.parquet       ← ratings_raw.parquet
    dim_movies_clean.parquet         ← movies_raw.parquet
    dim_tags_clean.parquet           ← tags_raw.parquet
    dim_links_clean.parquet          ← links_raw.parquet
    dim_genome_tags_clean.parquet    ← genome_tags_raw.parquet (optional)
    fact_genome_scores_clean.parquet ← genome_scores_raw.parquet (optional)

PHASE 3 — TEMPORAL SPLIT (Leakage-Safe Evaluation Scaffold)
  tables/
    interactions_train.parquet  ← fact_ratings_clean.parquet
    interactions_test.parquet   ← fact_ratings_clean.parquet

PHASE 4 — FEATURE ENGINEERING & TRANSACTION DATABASE (train-only)
  tables/
    user_features_train.parquet   ← interactions_train + dim_movies_clean (+ tags filtered by cutoff)
    movie_features_train.parquet  ← interactions_train + dim_movies_clean (+ tags filtered by cutoff)
    transactions_train.parquet    ← interactions_train (likes) + dim_movies_clean (genre tokens) + optional tag tokens

PHASE 4.5 — REDUCTION (Story B.1 prep)
  tables/
    transactions_train_reduced.parquet  ← transactions_train.parquet (vocab + basket pruning)
    token_stats.parquet                ← token df before vs after reduction
  reports/
    reduction_manifest.json            ← reduction knobs + before/after counts
    basket_stats.json                  ← basket stats before/after
    token_stats.csv                    ← human-readable token stats
Four proposed story modules (Part B):
Story A — Taste Tribes: cluster users into interpretable preference segments (train-only user profiles).
Story B.1 — Basket Liked-Movie Rules: mine frequent itemsets and association rules from “liked-movie” baskets.
Story B.2 — Anti-Preference Rules: mine rules for items users are unlikely to like.
Story B.3 — Temporal Shift Patterns: analyze changes in user preferences over time.
Story C — Behavioral Weirdness: detect unusual users and polarizing movies using anomaly signals.
Story D — Rating Foresight: predict ratings or “like” labels and analyze success/failure slices.
In short,
B1 = build baskets from “liked”
B2 = build baskets from “disliked” (same machinery, different filter)
B3 = build baskets from “liked/disliked” per time window + compare rules across windows
For details, please refer to the Story modules section.

A quick introduction to the Parquet tables
Each Parquet file is like a database table: fixed schema, efficient scans/joins, and friendly for analytics. The story modules primarily use the derived Mining-Ready layer.

Clean contract tables (foundation)
Parquet table	Built from	What it is for (short)
dim_movies_clean.parquet	movies.csv → movies_raw.parquet	Movie catalog and metadata; interpret results; generate genre features/tokens.
fact_ratings_clean.parquet	ratings.csv → ratings_raw.parquet	Canonical rating event log; main interaction history for split + modeling.
dim_tags_clean.parquet	tags.csv → tags_raw.parquet	Canonical tag event log with normalized tags; enrich features and explainability.
dim_links_clean.parquet	links.csv → links_raw.parquet	External identifiers (IMDb/TMDb) for enrichment and nicer dashboards.
dim_genome_tags_clean.parquet (optional)	genome-tags.csv → genome_tags_raw.parquet	Genome tag dictionary (tagId → text).
fact_genome_scores_clean.parquet (optional)	genome-scores.csv → genome_scores_raw.parquet	Genome relevance matrix; optional content-based representations/similarity.
Derived tables (Mining-Ready)
Parquet table	Built from	What it is for (short)
interactions_train.parquet	fact_ratings_clean.parquet	Training-period ratings (time ≤ cutoff). Used to build features, baskets, and models without leakage.
interactions_test.parquet	fact_ratings_clean.parquet	Held-out later ratings (time > cutoff). Used for sanity checks and evaluation scaffolding.
user_features_train.parquet	interactions_train + dim_movies_clean (+ tags filtered by cutoff)	Per-user feature vectors for clustering and outlier detection (train-only).
movie_features_train.parquet	interactions_train + dim_movies_clean (+ tags filtered by cutoff)	Per-movie feature vectors for popularity, entropy, polarization, genre flags, and tag summaries (train-only).
transactions_train.parquet	interactions_train (likes) + dim_movies_clean (genre tokens) + optional tag tokens	Transaction database (basket/itemset) for frequent itemsets and association rules.
transactions_train_reduced.parquet	transactions_train.parquet (Phase 4.5 Reduction)	Mining-optimized baskets for Story B.1 (vocab-pruned + basket-pruned).
⛏️📜 Story modules: Data consumption and mining methods guidelines
Each story module should read Mining-Ready Parquet inputs, run a coherent mining workflow, and produce both results and a concise narrative. The four proposed story modules below are starting points; you can extend or propose alternatives.

Each story module should produce both data outputs (e.g., tables of cluster labels, rules, anomaly scores, predictions) and an interpretation layer (e.g., markdown narrative, visualizations) that explains the key findings in human-readable form. See Appendix 2: Story module outputs + interpretation layer (A/B/C/D) for more details on expected outputs and interpretation guidelines.

📜Story A — Taste Tribes (Clustering / Segmentation)
Goal: group users into interpretable preference segments (“taste tribes”).

Primary inputs (Parquet):

user_features_train.parquet (main feature matrix)
dim_movies_clean.parquet (decode genres/titles for interpretation)
interactions_train.parquet / interactions_test.parquet (optional: sanity checks)
Suggested methods:

Feature prep: standardization (z-score), optional log transforms, missing-value handling.
Optional DR: PCA (diagnostics), UMAP (visualization).
Clustering baselines: K-Means, GMM, hierarchical clustering.
Evaluation: silhouette, Davies–Bouldin, size balance, interpretability checks.
Typical outputs:

cluster_labels_users.parquet (userId → cluster_id)
cluster_profiles.parquet / .json (centroids/summary stats)
“Cluster cards” (markdown): descriptions + examples
📜Story B.1 — Basket Liked-Movie Rules (Frequent Patterns / Association Rules)
Goal: discover frequent co-preference patterns and interpretable rules from “liked-movie” baskets.

Current implementation (Feb 2026): the Story B.1 module ships 3 presets (A/B/C) with timestamped runs. Outputs are written under movielens-story-modules-b/data/story-module-outputs/story_b.1/<preset>/<run_id>/ (current preset folders: preset_A_movie_only_full, preset_B_controlled_movie_tag_full, preset_C_genre_tag_full).

Primary inputs (Parquet):

transactions_train_reduced.parquet (preferred: mining-optimized baskets)
dim_movies_clean.parquet (decode movieId/genre tokens into titles/genres)
interactions_test.parquet (optional: persistence check of discovered rules)
Companion artifacts (reproducibility):

reports/reduction_manifest.json (all reduction knobs + before/after counts)
reports/basket_stats.json (basket-level before/after summary)
tables/token_stats.parquet / reports/token_stats.csv (token df before vs after)
Suggested methods:

Frequent itemset mining: FP-Growth (preferred), Apriori (baseline), Eclat (optional).
3-tier mining strategy: Tier 1 genre-only (min_support ≈ 0.05) → Tier 2 genre + tags (0.02) → Tier 3 full vocab (0.01). Start tight; loosen progressively.
Association rules: rank/filter by support, confidence, lift (optional: leverage/conviction).
Pruning: min support, max itemset size, remove redundancy, keep human-readable tokens.
Rule-based recommender: fire matching rules per user, aggregate scores, rank unseen candidates. Evaluate with Precision@K / NDCG@K (K ∈ {5, 10, 20}); baselines: MostPop, ContentCosine (via genre vectors).
Typical outputs:

frequent_itemsets.parquet (itemset, support)
association_rules.parquet (antecedent, consequent, support, confidence, lift, …)
predictions_test.parquet (userId, movieId, rank, method) + metrics.json (Precision@K, NDCG@K per method)
A short narrative: strongest rules, examples, and chosen filters/thresholds
📜Story B.2 — Anti-Preference Rules (negative taste / avoidance patterns)
Goal: mine frequent patterns and rules from disliked items to discover “taste boundaries” (what users tend to reject together).

Primary inputs (Parquet):

interactions_train.parquet (train-only interactions with rating + timestamp)
dim_movies_clean.parquet (decode movieId tokens into titles/genres)
Transaction definition:

Per-user basket built from disliked movies (e.g., rating ≤ 2.0 or ≤ 2.5).
Tokens: movie:<movieId> (optionally add genres/tags, but keep readability).
Mining target: frequent itemsets + association rules among disliked items.

Typical outputs:

anti_transactions_train.parquet (userId → disliked-item basket)
anti_rules.parquet (antecedent, consequent, support, confidence, lift)
Optional: “avoidance recommendations” (e.g., “If you disliked X, you might also dislike Y”) to filter or explain candidates
Short narrative: strongest avoidance rules + what segments they imply
📜Story B.3 — Temporal Shift Patterns (taste drift over time)
Goal: mine patterns in time-sliced baskets and compare windows to detect stable vs shifting preferences.

Primary inputs (Parquet):

interactions_train.parquet (requires timestamp column such as rating_dt_utc)
dim_movies_clean.parquet (decode movieId/genre tokens)
Transaction definition (choose one):

For each user, create baskets per time window (e.g., quarterly/yearly) using liked events (rating ≥ threshold) within the window; or
Use all rated movies within the window and optionally add rating-binned tokens.
Mining target: mine frequent itemsets/rules within each window, then compare windows to find emerging, fading, and stable patterns.

Typical outputs:

transactions_by_window_train.parquet (userId, window → basket)
rules_by_window.parquet (rules with window metadata)
pattern_shift_report.parquet and/or pattern_shift_report.md (ranked deltas: Δsupport / Δlift)
Short narrative: what changed, what stayed stable, and why it may matter
Regarding preprocessed data
B2 (Anti-Preference Rules)
Can reuse: the same leakage-safe split + cleaned ratings you used for B1.
Needs a derived table: an anti-transactions table (users → “disliked” items/tokens).
Same schema as transactions_train.parquet, just built with a different label rule (e.g., rating ≤ 2, or below-user-mean, etc.).
So: not a new preprocessing stage, but yes, it’s a new derived artifact like transactions_train_disliked.parquet (or transactions_train_b2.parquet).
B3 (Temporal Shift Patterns)
Can reuse: the same cleaned ratings + timestamps + split.
Needs derived tables: either
one transactions table with a window_id column, or
separate transactions per window (e.g., transactions_train_w01.parquet, ..._w02.parquet).
The extra work is windowing logic, not data cleaning.
📜Story C — Behavioral Weirdness (Outlier Detection + Polarization)
Goal: identify unusual users and movies with extreme/polarized rating behavior.

Primary inputs (Parquet):

user_features_train.parquet (user anomaly signals)
movie_features_train.parquet (movie polarization/outlier signals)
interactions_train.parquet / interactions_test.parquet (evidence slices + persistence checks)
dim_movies_clean.parquet (titles/genres for interpretation)
Suggested methods:

User outliers: Isolation Forest, LOF, One-Class SVM (optional), robust z-score baselines.
Movie polarization: rank by combinations of std, entropy, polarization proxies; optional simple mixtures.
Validation: check persistence in interactions_test and produce evidence plots.
Typical outputs:

user_anomaly_scores.parquet (userId → score, rank, method)
movie_anomaly_scores.parquet (movieId → score, rank, rationale features)
Short case studies (markdown): “what makes this user/movie weird?”
📜Story D — Rating Foresight (Supervised Learning: Prediction)
Goal: turn MovieLens into a supervised prediction task: predict a user’s rating for a movie (regression) or predict “like” vs “not-like” (classification), then explain where the model succeeds/fails.

Primary inputs (Parquet):

interactions_train.parquet and interactions_test.parquet (leakage-safe train/test interaction logs)
user_features_train.parquet (recommended: user-side predictors)
movie_features_train.parquet (recommended: movie-side predictors)
dim_movies_clean.parquet (optional: labels/genres for slicing & interpretation)
dim_tags_clean.parquet (optional: tag-based signals)
dim_genome_tags_clean.parquet / fact_genome_scores_clean.parquet (optional: stronger content signals)
Suggested methods:

Baselines: global mean rating; user mean; movie mean; (user + movie) bias model.
Regression: linear regression (interpretable) or gradient boosting / random forest (strong tabular baseline).
Classification: logistic regression (interpretable) or gradient boosting / random forest (strong baseline).
Evaluation: RMSE/MAE (regression) or AUC/F1 (classification), plus residual/error slicing (cold-start, popularity, activity).
Typical outputs:

Predictions table (e.g., predictions_test.parquet): userId, movieId, y_true, y_pred, error, timestamp.
Model card / run report: algorithm, hyperparameters, seed, feature set, train/test sizes.
Feature importance / coefficients (where supported) + short narrative: “what signals predict liking?” and “where do we fail?”
Cross-story connections
The four stories illuminate each other. Consider these cross-story data flows when planning advanced experiments:

Connection	How
A → B	Add cluster labels as meta-tokens in Story B.1 baskets (e.g., cluster:action_fans). Rules then reveal co-preferences specific to a taste tribe.
A → C	Overlay anomaly scores from Story C onto the cluster map. Anomalous users may sit at cluster boundaries — visual evidence of behavioral outliers.
A → D	Add cluster_id as a feature in Story D’s modeling frame. Genuine taste clusters should reduce prediction error.
B → C	Users who trigger contradictory high-lift rules (e.g., both horror-hater and horror-lover rules fire) are candidates for behavioral weirdness.
C → B	Remove anomalous users from the basket corpus and re-mine. If rules change significantly, original rules may have been driven by outlier behavior.
D → A	Story D error slices by cluster reveal which taste tribes are hardest to predict — those clusters may have more internal diversity.
Implementation priority & suggested timeline
Story	Effort	Priority	Rationale
B1 — Basket Liked-Movie Rules	Low–Medium	1st	Data is already basket-shaped and reduced. Clear metrics (support/confidence/lift/NDCG). Main work: FP-Growth + rule-based recommender.
A — Taste Tribes	Medium	2nd	K-Means (or similar) on the feature matrix is straightforward. Main work: K-selection, cluster interpretation, exemplar movie extraction.
B2 — Anti-Preference Rules	Low–Medium	3rd	Same mining machinery as B1, but with a “dislike” basket definition. Main work: define dislike threshold, build anti-transactions, mine/prune readable avoidance rules.
B3 — Temporal Shift Patterns	Medium	4th	Adds time windowing + rule comparison across windows. Main work: window design, mining per window, and delta reports (Δsupport/Δlift) for emerging/fading/stable patterns.
D — Rating Foresight	Medium–High	5th	Requires building a large modeling frame. Baselines are simple; learned models need tuning. Error slicing is the key deliverable (cold-start, popularity, activity).
C — Weirdness	High	6th	No ground truth — requires careful definition of anomalous, persistence/injection testing, and convincing case studies with evidence slices.
Week	Focus
1	Story B.1: FP-Growth + rules + recommender + ranking evaluation
2	Story A: clustering + K-selection + cluster cards
3	Story B.2/B.3: anti-preference baskets + temporal windowing scaffolding + comparative rule summaries
4	Story C: anomaly detection + polarization + case studies
5	Story D (optional) + cross-story connections + final report synthesis
Appendix 1: How derived tables (mining-ready data) are generated
The key derived tables powering story modules are built primarily from fact_ratings_clean.parquet. Any table used to learn patterns/models is generated from training-period data only.

fact_ratings_clean + (dim_movies_clean, dim_tags_clean)
        └─(temporal cutoff)→ interactions_train / interactions_test
          ├─(aggregate)→ user_features_train (+ movie_features_train)
          └─(like filter + tokenize + group)→ transactions_train
            └─(vocab + basket pruning)→ transactions_train_reduced
A) interactions_train and interactions_test
Core idea: choose a cutoff datetime cutoff_dt and split ratings by time. Train uses earlier interactions; test uses later interactions.

interactions_train: rows where rating_dt_utc < cutoff_dt
interactions_test: rows where rating_dt_utc ≥ cutoff_dt
Why temporal (not random) split? MovieLens is timestamped; temporal splitting avoids subtle “time travel” leakage.

B) user_features_train (and movie_features_train)
Core idea: aggregate training interactions into one row per entity (user/movie) to form feature vectors. These are the primary inputs for clustering and outlier workflows.

Activity: counts, active days, time span.
Rating behavior: mean, variance/std, extremes.
Preferences: genre-based signals via joins with dim_movies_clean.
Optional semantics: tag usage/diversity (train-period tags only).
C) transactions_train
Core idea: convert training interactions into basket-style transactions for frequent pattern mining.

Like filter: keep “liked” events using a threshold like_threshold (e.g., rating ≥ threshold).
Tokenize: represent each liked movie as movie:<movieId>.
Optional enrichment: add genre tokens (genre:Comedy) and tag tokens (tag:time_travel).
Group: group by userId to produce one basket per user.
Key knobs: like_threshold controls basket density; token inclusion rules (genres/tags) affect rule readability.

D) transactions_train_reduced (Phase 4.5 Reduction)
Core idea: apply vocabulary pruning and basket filtering to keep Story B.1 mining feasible and reproducible. The reduced table keeps the same conceptual schema (one basket per user), but with fewer tokens and/or fewer baskets.

Vocabulary pruning: token-family filtering (genre/movie/tag), minimum token df, optional top-K caps.
Basket pruning: drop baskets that become too small; optionally cap basket size.
Optional dev-only sampling: sample a fraction of baskets for faster iteration.
Sample rows (quick preview)
Below are a few illustrative sample rows for derived tables.

Show 3 sample rows per derived table
Appendix 2: Story module outputs + interpretation layer (A/B/C/D)
This appendix defines the output contract for story modules: what each module should write as machine-readable artifacts (primarily Parquet + JSON) and what it should ship as the interpretation layer (narrative text + visuals).

0) Two-layer output philosophy
Each story module produces outputs in two layers:

Core artifacts (machine-readable)
Purpose: enable re-use, comparison, and downstream visualization.
Typical formats: *.parquet for tables; *.json for metrics/config/provenance; optionally *.pkl / *.joblib for serialized models.
Interpretation layer (human-readable)
Purpose: explain what the artifact means and why it matters.
Typical formats: *.md narrative (“cards”, “case studies”, “model card”), plus figures (*.png / *.svg).
Rule of thumb: Parquet is the contract; markdown/figures are the explanation.

1) Where outputs should live (recommended)
Core pipeline outputs (inputs to stories)
Stories should read (only) from data/processed/tables/*.parquet.
Use data/processed/reports/*.json for audit/split metadata.
Story module outputs
Story modules must not mutate data/processed/.... Write outputs under a story namespace.

artifacts/
  story_A/
  story_B/
  story_C/
  story_D/
Each story folder should contain:

tables/ for Parquet outputs
reports/ for JSON + narrative markdown
figures/ for plots
artifacts/story_B/
  tables/association_rules.parquet
  reports/metrics.json
  reports/summary.md
  figures/top_rules.png
Story B.1 (as implemented in this project): outputs are organized by preset and run id:

movielens-story-modules-b/
  data/story-module-outputs/story_b/
    preset_A_movie_only_full/<run_id>/
      tables/
      reports/
      figures/
    preset_B_controlled_movie_tag_full/<run_id>/
      tables/
      reports/
      figures/
    preset_C_genre_tag_full/<run_id>/
      tables/
      reports/
      figures/
The Streamlit demo app (movielens-story-modules-b/demo_apps/story_b_explainable_recommender) auto-discovers the latest run_id inside each preset directory. Note: the app’s current preset directory constants are preset_A_movie_only, preset_B_controlled_movie_tag, preset_C_genre_tag (no _full suffix). If your outputs use *_full, update the app constants or rename the preset folders.

2) Required provenance (applies to all stories)
Every story module should write:

reports/run_manifest.json with: inputs (optional row counts), method name(s), key parameters (thresholds/k/seed), code/notebook identifier (optional), timestamp.
reports/summary.md (5–15 lines: what you did, what you found, what’s uncertain).
3) Story A — Taste Tribes (Clustering / Segmentation)
Core artifacts:

tables/cluster_labels_users.parquet (minimum columns: userId, cluster_id, method) — method stores the algorithm name (e.g., "kmeans_k8") for multi-run comparison
tables/cluster_profiles.parquet (or reports/cluster_profiles.json) with centroids/summary stats per cluster
Optional: tables/user_embedding_2d.parquet (if using PCA/UMAP) with userId, x, y (and optional cluster_id)
Interpretation layer:

reports/cluster_cards.md (descriptions + examples)
figures/cluster_map.png, figures/cluster_profiles.png
4) Story B.1 — Basket Liked-Movie Rules (Frequent Patterns / Association Rules)
Core artifacts:

tables/frequent_itemsets.parquet (minimum: itemset, support) — itemset is a Parquet list of token strings (e.g., ["movie:318", "genre:drama"])
tables/association_rules.parquet (minimum: antecedent, consequent, support, confidence, lift)
Optional: tables/rules_human_readable.parquet (decoded tokens joined to titles/genres)
Interpretation layer:

reports/summary.md (thresholds + pruning + strongest coherent rules)
reports/eval_metrics.json + reports/eval_summary.md (held-out evaluation; typically only for movie-only runs)
figures/rule_support_vs_lift.png, figures/top_rules_table.png
Optional variants (Story B.2 / Story B.3) (same pattern-mining spirit):

Story B.2 — Anti-Preference Rules: tables/anti_transactions_train.parquet, tables/anti_rules.parquet, plus a short reports/summary.md of “avoidance” patterns.
Story B.3 — Temporal Shift Patterns: tables/transactions_by_window_train.parquet, tables/rules_by_window.parquet, and reports/pattern_shift_report.md (or tables/pattern_shift_report.parquet) ranking Δsupport/Δlift.
5) Story C — Behavioral Weirdness (Outlier Detection + Polarization)
Core artifacts:

tables/user_anomaly_scores.parquet (minimum: userId, score, rank, method)
tables/movie_anomaly_scores.parquet (minimum: movieId, score, rank + rationale features)
Optional: tables/polarizing_movies.parquet (if separated from anomaly scoring)
Interpretation layer:

reports/case_studies.md (3–10 examples with evidence slices)
figures/user_anomaly_scatter.png, figures/movie_polarization_histograms/…
6) Story D — Rating Foresight (Supervised Learning: Prediction)
Core artifacts:

tables/predictions_test.parquet (minimum: userId, movieId, y_true, y_pred, error, timestamp) — regression: floats (0.5–5.0); classification: integers (0/1)
reports/metrics.json (RMSE/MAE or AUC/F1 + dataset sizes + evaluation split) — must include at least one baseline (e.g., rmse_baseline_global_mean, rmse_baseline_user_item_bias) for comparison
reports/model_card.md (algorithm, hyperparameters, seed, feature set, train/test sizes)
Optional: models/model.joblib, tables/feature_importance.parquet
Interpretation layer:

reports/error_slices.md (cold-start, popularity bins, user activity bins)
figures/residuals.png or figures/roc_curve.png / figures/pr_curve.png
7) Minimal “Definition of Done” checklist per story
At least one primary Parquet table artifact (labels/rules/scores/predictions)
At least one JSON metrics/config artifact (or a run manifest)
At least one short markdown narrative (summary / cards / case studies / model card)
At least one figure that supports the narrative
Provenance includes inputs, key parameters, and timestamp
8) Quick cross-story naming alignment
K eep primary keys (userId, movieId) intact.
Use train / test suffixes when artifacts are split-dependent.
Prefer Parquet for any table you might join/aggregate later.
Appendix 3: Naming Conventions and Data Scale
Naming conventions (how to read file names)
Pattern	Meaning
_clean	validated, contract-compliant canonical table (Phase 2)
_train	derived from training data only; leakage-safe
_test	held-out evaluation partition
dim_	dimension table: lookup/metadata (movies, tags, links, genome tags)
fact_	fact table: event/measurement log (ratings, genome scores)
Data scale
Metric	Value
Total ratings	33,832,162
Users (total)	330,975
Movies (with ratings)	86,537
Training ratings (90.2%)	30,508,831
Test ratings (9.8%)	3,323,331
Unique users in train	306,595
Unique movies in train	64,717
Transaction baskets (reduced)	300,137
Vocab after reduction (55,400 → 5,519 tokens)	5,519
Temporal cutoff: 2020-09-11. Cold-start users in test ≈ 74.9%. Rating distribution stable across split (train mean 3.54, test mean 3.57, KL = 0.029).

Page created by Nguyễn Sỹ Hùng for the Data Mining Course's Group Final Project. The content is a practical proposal and guide for story module development, not official documentation. Each team member is encouraged to adapt and extend the ideas as contributions.