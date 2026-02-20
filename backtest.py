"""10-Year Backtest for Box Office Scoring Algorithm (2015-2024).

Validates the scoring algorithm in scoring.py against actual box office results.
For each year, predicts which movies would score highest using only pre-year data,
then compares predicted rankings against actual revenue rankings.

Usage:
    python backtest.py
"""

import json
import os
import sqlite3
import sys
import time

import requests
from dotenv import load_dotenv

from scoring import (
    WEIGHT_CAST,
    WEIGHT_DIRECTOR,
    WEIGHT_FRANCHISE,
    WEIGHT_PRODUCER,
    compute_person_score,
)

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
DB_PATH = os.path.join(os.path.dirname(__file__), "backtest.db")
REQUEST_DELAY = 0.05  # 50ms → ~20 req/sec


# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------

def api_get(endpoint, params=None):
    """TMDB GET with rate limiting."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    time.sleep(REQUEST_DELAY)
    resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Cache DB setup
# ---------------------------------------------------------------------------

def init_cache_db():
    """Create the backtest cache database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS movie_cache (
            tmdb_id         INTEGER PRIMARY KEY,
            title           TEXT,
            release_date    TEXT,
            budget          INTEGER,
            revenue         INTEGER,
            belongs_to_collection TEXT
        );
        CREATE TABLE IF NOT EXISTS person_filmography_cache (
            tmdb_person_id  INTEGER PRIMARY KEY,
            name            TEXT,
            filmography_json TEXT
        );
        CREATE TABLE IF NOT EXISTS credits_cache (
            tmdb_movie_id   INTEGER PRIMARY KEY,
            credits_json    TEXT
        );
        CREATE TABLE IF NOT EXISTS backtest_results (
            year            INTEGER PRIMARY KEY,
            n_movies        INTEGER,
            rho             REAL,
            overlap_10      INTEGER,
            overlap_20      INTEGER,
            profit_score    REAL,
            profit_optimal  REAL,
            profit_random   REAL,
            dr_ratio        REAL
        );
        CREATE TABLE IF NOT EXISTS backtest_movies (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            year            INTEGER NOT NULL,
            tmdb_id         INTEGER NOT NULL,
            title           TEXT,
            budget          INTEGER,
            revenue         INTEGER,
            predicted_score REAL,
            predicted_rank  INTEGER,
            actual_rank     INTEGER,
            UNIQUE(year, tmdb_id)
        );
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Cached fetchers
# ---------------------------------------------------------------------------

def get_movie_details(conn, tmdb_id):
    """Return movie dict from cache or TMDB API."""
    row = conn.execute(
        "SELECT tmdb_id, title, release_date, budget, revenue, belongs_to_collection "
        "FROM movie_cache WHERE tmdb_id = ?", (tmdb_id,)
    ).fetchone()
    if row:
        return {
            "tmdb_id": row[0], "title": row[1], "release_date": row[2],
            "budget": row[3], "revenue": row[4], "belongs_to_collection": row[5],
        }
    data = api_get(f"/movie/{tmdb_id}")
    coll = json.dumps(data["belongs_to_collection"]) if data.get("belongs_to_collection") else None
    conn.execute(
        "INSERT OR IGNORE INTO movie_cache (tmdb_id, title, release_date, budget, revenue, belongs_to_collection) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (tmdb_id, data.get("title", ""), data.get("release_date", ""),
         data.get("budget", 0), data.get("revenue", 0), coll),
    )
    conn.commit()
    return {
        "tmdb_id": tmdb_id, "title": data.get("title", ""),
        "release_date": data.get("release_date", ""),
        "budget": data.get("budget", 0), "revenue": data.get("revenue", 0),
        "belongs_to_collection": coll,
    }


def get_credits(conn, tmdb_movie_id):
    """Return credits dict {directors, producers, cast} from cache or API."""
    row = conn.execute(
        "SELECT credits_json FROM credits_cache WHERE tmdb_movie_id = ?",
        (tmdb_movie_id,)
    ).fetchone()
    if row:
        return json.loads(row[0])

    data = api_get(f"/movie/{tmdb_movie_id}/credits")
    cast_list = data.get("cast", [])
    crew_list = data.get("crew", [])

    directors = [
        {"id": c["id"], "name": c["name"]}
        for c in crew_list if c.get("job") == "Director"
    ]
    producers = []
    seen = set()
    for c in crew_list:
        if c.get("job") in ("Producer", "Executive Producer") and c["id"] not in seen:
            seen.add(c["id"])
            producers.append({"id": c["id"], "name": c["name"]})
            if len(producers) >= 3:
                break
    cast = [
        {"id": c["id"], "name": c["name"]}
        for c in cast_list[:5]
    ]

    credits = {"directors": directors, "producers": producers, "cast": cast}
    conn.execute(
        "INSERT OR IGNORE INTO credits_cache (tmdb_movie_id, credits_json) VALUES (?, ?)",
        (tmdb_movie_id, json.dumps(credits)),
    )
    conn.commit()
    return credits


def get_filmography(conn, person_tmdb_id):
    """Return full filmography list from cache or API."""
    row = conn.execute(
        "SELECT filmography_json FROM person_filmography_cache WHERE tmdb_person_id = ?",
        (person_tmdb_id,)
    ).fetchone()
    if row:
        return json.loads(row[0])

    data = api_get(f"/person/{person_tmdb_id}/movie_credits")
    all_movies = []
    for m in data.get("cast", []):
        if m.get("release_date"):
            all_movies.append({"id": m["id"], "title": m.get("title", ""),
                               "release_date": m["release_date"]})
    for m in data.get("crew", []):
        if m.get("release_date") and m.get("job") in ("Director", "Producer", "Executive Producer"):
            all_movies.append({"id": m["id"], "title": m.get("title", ""),
                               "release_date": m["release_date"]})

    # Deduplicate
    seen = set()
    unique = []
    for m in all_movies:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)

    conn.execute(
        "INSERT OR IGNORE INTO person_filmography_cache (tmdb_person_id, name, filmography_json) "
        "VALUES (?, ?, ?)",
        (person_tmdb_id, "", json.dumps(unique)),
    )
    conn.commit()
    return unique


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_person_for_year(conn, person_tmdb_id, cutoff_date):
    """Score a person using only movies released before cutoff_date."""
    filmography = get_filmography(conn, person_tmdb_id)

    # Temporal filter: only movies before cutoff
    eligible = [
        m for m in filmography
        if m.get("release_date", "") < cutoff_date and m.get("release_date", "") > "2000-01-01"
    ]
    eligible.sort(key=lambda x: x.get("release_date", ""), reverse=True)
    recent = eligible[:3]

    history_rows = []
    for m in recent:
        details = get_movie_details(conn, m["id"])
        budget = details["budget"]
        revenue = details["revenue"]
        profit = revenue - budget if (budget > 0 and revenue > 0) else 0
        history_rows.append((
            details["title"], details["release_date"],
            budget, revenue, profit
        ))

    score, _ = compute_person_score(history_rows)
    return score


def score_movie_for_year(conn, tmdb_id, year):
    """Compute the full predicted score for a movie using pre-year data only."""
    cutoff_date = f"{year}-01-01"
    credits = get_credits(conn, tmdb_id)

    # Director score (max)
    dir_scores = [
        score_person_for_year(conn, d["id"], cutoff_date)
        for d in credits["directors"]
    ]
    dir_score = max(dir_scores) if dir_scores else 0.0

    # Producer score (max)
    prod_scores = [
        score_person_for_year(conn, p["id"], cutoff_date)
        for p in credits["producers"]
    ]
    prod_score = max(prod_scores) if prod_scores else 0.0

    # Cast score (avg of top 3)
    cast_scores = [
        score_person_for_year(conn, c["id"], cutoff_date)
        for c in credits["cast"]
    ]
    cast_avg = sum(cast_scores[:3]) / len(cast_scores[:3]) if cast_scores else 0.0

    # Franchise bonus
    movie = get_movie_details(conn, tmdb_id)
    franchise_score = 50.0 if movie.get("belongs_to_collection") else 0.0

    total = (
        dir_score * WEIGHT_DIRECTOR
        + prod_score * WEIGHT_PRODUCER
        + cast_avg * WEIGHT_CAST
        + franchise_score * WEIGHT_FRANCHISE
    )
    return total


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_top_movies(conn, year, n=50):
    """Discover top n movies from `year` by revenue via TMDB discover endpoint.

    Returns list of dicts with tmdb_id, title, revenue, budget.
    Only includes movies with budget > 0 and revenue > 0.
    """
    movies = []
    page = 1
    while len(movies) < n:
        data = api_get("/discover/movie", {
            "primary_release_date.gte": f"{year}-01-01",
            "primary_release_date.lte": f"{year}-12-31",
            "sort_by": "revenue.desc",
            "page": page,
        })
        results = data.get("results", [])
        if not results:
            break
        for m in results:
            details = get_movie_details(conn, m["id"])
            if details["budget"] > 0 and details["revenue"] > 0:
                movies.append(details)
                if len(movies) >= n:
                    break
        page += 1
        if page > data.get("total_pages", 1):
            break
    return movies


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman_rho(predicted_ranks, actual_ranks):
    """Compute Spearman rank correlation coefficient.

    Both inputs are lists of the same length where index i corresponds to movie i.
    predicted_ranks[i] = rank of movie i by predicted score (1 = best).
    actual_ranks[i]    = rank of movie i by actual revenue (1 = best).
    """
    n = len(predicted_ranks)
    if n < 2:
        return 0.0
    d_squared_sum = sum(
        (p - a) ** 2 for p, a in zip(predicted_ranks, actual_ranks)
    )
    return 1.0 - 6.0 * d_squared_sum / (n * (n ** 2 - 1))


def top_k_overlap(predicted_top_k_ids, actual_top_k_ids):
    """Count how many IDs appear in both sets."""
    return len(set(predicted_top_k_ids) & set(actual_top_k_ids))


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def backtest_year(conn, year):
    """Run the backtest for a single year. Returns a result dict."""
    print(f"\n{'='*60}")
    print(f"  YEAR {year}")
    print(f"{'='*60}")

    # 1. Discover top 50 movies
    print(f"  Discovering top 50 movies by revenue...")
    movies = discover_top_movies(conn, year, n=50)
    if not movies:
        print(f"  No movies found for {year}, skipping.")
        return None
    print(f"  Found {len(movies)} movies with budget & revenue > 0")

    # 2. Score each movie
    print(f"  Scoring movies (fetching credits & filmographies)...")
    scored = []
    for i, m in enumerate(movies):
        try:
            score = score_movie_for_year(conn, m["tmdb_id"], year)
        except Exception as e:
            print(f"    Warning: Could not score '{m['title']}': {e}")
            score = 0.0
        scored.append({
            "tmdb_id": m["tmdb_id"],
            "title": m["title"],
            "revenue": m["revenue"],
            "budget": m["budget"],
            "score": score,
        })
        if (i + 1) % 10 == 0:
            print(f"    Scored {i + 1}/{len(movies)}")

    # 3. Rank by predicted score and by actual revenue
    by_score = sorted(scored, key=lambda x: x["score"], reverse=True)
    by_revenue = sorted(scored, key=lambda x: x["revenue"], reverse=True)

    # Build rank maps (1-indexed)
    score_rank = {m["tmdb_id"]: rank + 1 for rank, m in enumerate(by_score)}
    revenue_rank = {m["tmdb_id"]: rank + 1 for rank, m in enumerate(by_revenue)}

    all_ids = [m["tmdb_id"] for m in scored]
    pred_ranks = [score_rank[tid] for tid in all_ids]
    act_ranks = [revenue_rank[tid] for tid in all_ids]

    # 4. Compute metrics
    rho = spearman_rho(pred_ranks, act_ranks)

    top10_pred_ids = [m["tmdb_id"] for m in by_score[:10]]
    top10_act_ids = [m["tmdb_id"] for m in by_revenue[:10]]
    top20_pred_ids = [m["tmdb_id"] for m in by_score[:20]]
    top20_act_ids = [m["tmdb_id"] for m in by_revenue[:20]]

    overlap_10 = top_k_overlap(top10_pred_ids, top10_act_ids)
    overlap_20 = top_k_overlap(top20_pred_ids, top20_act_ids)

    # 5. Draft profit simulation
    # "Profit" = sum of (revenue - budget) for drafted movies
    revenue_map = {m["tmdb_id"]: m for m in scored}

    def draft_profit(ids):
        return sum(revenue_map[tid]["revenue"] - revenue_map[tid]["budget"] for tid in ids)

    profit_score_10 = draft_profit(top10_pred_ids)
    profit_optimal_10 = draft_profit(top10_act_ids)

    # Random baseline: average profit of any 10 movies (use mean of all)
    all_profits = [m["revenue"] - m["budget"] for m in scored]
    avg_profit_per_movie = sum(all_profits) / len(all_profits) if all_profits else 0
    profit_random_10 = avg_profit_per_movie * 10

    # D/R ratio: drafted profit / random profit
    dr_ratio = profit_score_10 / profit_random_10 if profit_random_10 > 0 else 0.0

    # 6. Print per-year table
    print(f"\n  Spearman rho: {rho:.3f}")
    print(f"  Top-10 overlap: {overlap_10}/10    Top-20 overlap: {overlap_20}/20")
    print(f"  Draft profit (score top-10):   ${profit_score_10 / 1e6:,.0f}M")
    print(f"  Draft profit (optimal top-10): ${profit_optimal_10 / 1e6:,.0f}M")
    print(f"  Draft profit (random 10):      ${profit_random_10 / 1e6:,.0f}M")
    print(f"  D/R ratio: {dr_ratio:.2f}x")

    print(f"\n  {'Pred':>4}  {'Act':>4}  {'Score':>8}  {'Revenue':>12}  Title")
    print(f"  {'----':>4}  {'----':>4}  {'--------':>8}  {'------------':>12}  -----")
    for m in by_score[:10]:
        tid = m["tmdb_id"]
        print(f"  {score_rank[tid]:>4}  {revenue_rank[tid]:>4}  {m['score']:>8.1f}"
              f"  ${m['revenue'] / 1e6:>10,.0f}M  {m['title']}")

    # 7. Persist results to DB for web UI
    conn.execute("DELETE FROM backtest_results WHERE year = ?", (year,))
    conn.execute(
        "INSERT INTO backtest_results (year, n_movies, rho, overlap_10, overlap_20, "
        "profit_score, profit_optimal, profit_random, dr_ratio) VALUES (?,?,?,?,?,?,?,?,?)",
        (year, len(movies), rho, overlap_10, overlap_20,
         profit_score_10, profit_optimal_10, profit_random_10, dr_ratio),
    )
    conn.execute("DELETE FROM backtest_movies WHERE year = ?", (year,))
    for m in scored:
        tid = m["tmdb_id"]
        conn.execute(
            "INSERT INTO backtest_movies (year, tmdb_id, title, budget, revenue, "
            "predicted_score, predicted_rank, actual_rank) VALUES (?,?,?,?,?,?,?,?)",
            (year, tid, m["title"], m["budget"], m["revenue"],
             m["score"], score_rank[tid], revenue_rank[tid]),
        )
    conn.commit()

    return {
        "year": year,
        "n_movies": len(movies),
        "rho": rho,
        "overlap_10": overlap_10,
        "overlap_20": overlap_20,
        "profit_score": profit_score_10,
        "profit_optimal": profit_optimal_10,
        "profit_random": profit_random_10,
        "dr_ratio": dr_ratio,
    }


def print_summary(results):
    """Print the summary table across all years."""
    print(f"\n\n{'='*80}")
    print(f"  BACKTEST SUMMARY (2015-2025)")
    print(f"{'='*80}")
    print(f"  {'Year':>4}  {'N':>3}  {'Rho':>6}  {'T10':>3}  {'T20':>3}"
          f"  {'Score$M':>9}  {'Opt$M':>9}  {'Rand$M':>9}  {'D/R':>5}")
    print(f"  {'----':>4}  {'---':>3}  {'------':>6}  {'---':>3}  {'---':>3}"
          f"  {'---------':>9}  {'---------':>9}  {'---------':>9}  {'-----':>5}")

    sum_rho = 0
    sum_t10 = 0
    sum_t20 = 0
    sum_dr = 0
    count = 0

    for r in results:
        print(f"  {r['year']:>4}  {r['n_movies']:>3}  {r['rho']:>6.3f}  {r['overlap_10']:>3}"
              f"  {r['overlap_20']:>3}  {r['profit_score'] / 1e6:>8,.0f}"
              f"  {r['profit_optimal'] / 1e6:>8,.0f}"
              f"  {r['profit_random'] / 1e6:>8,.0f}  {r['dr_ratio']:>5.2f}")
        sum_rho += r["rho"]
        sum_t10 += r["overlap_10"]
        sum_t20 += r["overlap_20"]
        sum_dr += r["dr_ratio"]
        count += 1

    if count:
        print(f"  {'----':>4}  {'---':>3}  {'------':>6}  {'---':>3}  {'---':>3}"
              f"  {'---------':>9}  {'---------':>9}  {'---------':>9}  {'-----':>5}")
        print(f"  {'AVG':>4}  {'':>3}  {sum_rho / count:>6.3f}  {sum_t10 / count:>3.1f}"
              f"  {sum_t20 / count:>3.1f}  {'':>9}  {'':>9}  {'':>9}  {sum_dr / count:>5.2f}")

    print(f"\n  Interpretation guide:")
    print(f"    Rho  > 0.5   Strong positive correlation — algorithm captures revenue signal well")
    print(f"    Rho  0.3-0.5 Moderate correlation — useful but noisy")
    print(f"    Rho  < 0.3   Weak correlation — algorithm needs improvement")
    print(f"    D/R  > 1.5   Drafting by score yields 50%+ more profit than random")
    print(f"    D/R  ~ 1.0   No better than random draft")
    print(f"    T10  >= 5    Algorithm identifies at least half the actual top 10")


def main():
    if not API_KEY:
        print("Error: TMDB_API_KEY not set. Create a .env file with your API key.")
        print("Get a free key at https://www.themoviedb.org/settings/api")
        sys.exit(1)

    print("=" * 60)
    print("  BOX OFFICE SCORING ALGORITHM — 11-YEAR BACKTEST")
    print("  Validating against actual results from 2015-2025")
    print("=" * 60)

    conn = init_cache_db()
    results = []

    try:
        for year in range(2015, 2026):
            r = backtest_year(conn, year)
            if r:
                results.append(r)

        if results:
            print_summary(results)
        else:
            print("\nNo results to summarize.")

    finally:
        conn.close()

    print(f"\nCache saved to: {DB_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
