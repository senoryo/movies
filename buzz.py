"""Fetch internet buzz data for 2026 movies and compute buzz scores.

Searches the web for box-office predictions, critic reactions, and audience
hype signals for each movie, then stores an aggregated buzz_score in the
movie_buzz table.

Usage:
    python buzz.py
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

from duckduckgo_search import DDGS

DB_PATH = os.path.join(os.path.dirname(__file__), "movies.db")
BACKTEST_DB_PATH = os.path.join(os.path.dirname(__file__), "backtest.db")

# Search configuration
REQUEST_DELAY = 1.5  # seconds between searches to avoid rate limits
RETRY_DELAY = 4.0    # longer delay for retries to avoid 403s

# Sentiment keywords
POSITIVE_WORDS = {
    "blockbuster", "highly anticipated", "breakout", "hit", "smash",
    "exciting", "must-see", "promising", "huge", "massive", "eagerly awaited",
    "phenomenon", "record-breaking", "box office champion", "crowd-pleaser",
}
NEGATIVE_WORDS = {
    "flop", "bomb", "troubled", "disappointing", "disaster", "worry",
    "concerning", "delayed", "underperform", "lackluster", "struggling",
}


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def search_web(query, max_results=8):
    """Search DuckDuckGo and return result snippets."""
    try:
        results = DDGS().text(query, max_results=max_results)
        return [r["body"] for r in results if r.get("body")]
    except Exception as e:
        print(f"    Search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def extract_dollar_amounts(text):
    """Extract dollar amounts from text, returning values in millions."""
    amounts = []

    # Match patterns like "$150 million", "$1.2 billion", "$85M", "$150m"
    patterns = [
        r"\$\s*([\d,.]+)\s*billion",
        r"\$\s*([\d,.]+)\s*b\b",
        r"\$\s*([\d,.]+)\s*million",
        r"\$\s*([\d,.]+)\s*m\b",
        r"\$\s*([\d,.]+)\s*mil\b",
    ]

    lower = text.lower()
    for i, pat in enumerate(patterns):
        for match in re.finditer(pat, lower):
            try:
                val = float(match.group(1).replace(",", ""))
                # Convert billions to millions
                if i < 2:
                    val *= 1000
                amounts.append(val)
            except ValueError:
                continue

    return amounts


def compute_sentiment(text):
    """Compute sentiment score from -1.0 to 1.0 based on keyword matching."""
    lower = text.lower()
    pos_count = sum(1 for word in POSITIVE_WORDS if word in lower)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in lower)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def compute_buzz_signals(snippets):
    """Extract buzz signals from search result snippets.

    Returns dict with predicted_gross, source_count, sentiment, snippets.
    """
    all_text = " ".join(snippets)
    dollar_amounts = extract_dollar_amounts(all_text)

    # Use median predicted gross if we have multiple estimates
    predicted_gross = 0.0
    if dollar_amounts:
        sorted_amounts = sorted(dollar_amounts)
        mid = len(sorted_amounts) // 2
        predicted_gross = sorted_amounts[mid]

    source_count = len(snippets)
    sentiment = compute_sentiment(all_text)

    return {
        "predicted_gross": predicted_gross,
        "source_count": source_count,
        "sentiment": round(sentiment, 3),
        "snippets": snippets[:5],  # Store top 5 snippets for reference
    }


# ---------------------------------------------------------------------------
# Buzz score computation
# ---------------------------------------------------------------------------

def compute_raw_buzz_score(signals):
    """Compute raw buzz score from extracted signals.

    Primary signal: predicted gross (in millions)
    Secondary signals: source count (awareness proxy), sentiment

    Raw score is unnormalized — normalization happens across all movies.
    """
    gross = signals["predicted_gross"]
    source_count = signals["source_count"]
    sentiment = signals["sentiment"]

    # Primary signal: predicted gross (already in millions)
    score = gross

    # Source count multiplier: more sources = more awareness
    # Baseline is 5 sources, each additional adds 5% bonus
    if source_count > 0:
        awareness_mult = 0.8 + min(source_count, 10) * 0.04
        score *= awareness_mult

    # Sentiment adjustment: +/-20% max
    sentiment_mult = 1.0 + sentiment * 0.2
    score *= sentiment_mult

    # If no dollar prediction, use source count + sentiment as fallback
    if gross == 0 and source_count > 0:
        score = source_count * (1.0 + sentiment * 0.5) * 5.0

    return round(score, 2)


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def ensure_buzz_table(conn):
    """Create movie_buzz table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS movie_buzz (
            movie_id    INTEGER UNIQUE NOT NULL,
            buzz_score  REAL DEFAULT 0,
            raw_data    TEXT,
            fetched_at  TEXT,
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        )
    """)
    conn.commit()


def get_unfetched_movies(conn):
    """Return list of (id, title) for movies not yet in movie_buzz."""
    cursor = conn.execute("""
        SELECT m.id, m.title
        FROM movies m
        LEFT JOIN movie_buzz mb ON mb.movie_id = m.id
        WHERE mb.movie_id IS NULL
        ORDER BY m.id
    """)
    return cursor.fetchall()


def get_zero_source_movies(conn):
    """Return list of (id, title) for movies whose buzz search returned 0 sources."""
    cursor = conn.execute("""
        SELECT m.id, m.title
        FROM movies m
        JOIN movie_buzz mb ON mb.movie_id = m.id
        WHERE mb.raw_data IS NOT NULL
          AND json_extract(mb.raw_data, '$.source_count') = 0
        ORDER BY m.id
    """)
    return cursor.fetchall()


def store_buzz(conn, movie_id, buzz_score, raw_data):
    """Insert or update buzz data for a movie."""
    conn.execute("""
        INSERT OR REPLACE INTO movie_buzz (movie_id, buzz_score, raw_data, fetched_at)
        VALUES (?, ?, ?, ?)
    """, (
        movie_id,
        buzz_score,
        json.dumps(raw_data),
        datetime.now(timezone.utc).isoformat(),
    ))


def normalize_buzz_scores(conn):
    """Normalize all buzz scores to 0-100 scale."""
    cursor = conn.execute("SELECT MAX(buzz_score), MIN(buzz_score) FROM movie_buzz")
    max_score, min_score = cursor.fetchone()

    if max_score is None or max_score == min_score:
        conn.execute("UPDATE movie_buzz SET buzz_score = 0")
        conn.commit()
        return

    score_range = max_score - min_score
    conn.execute("""
        UPDATE movie_buzz
        SET buzz_score = ROUND(((buzz_score - ?) / ?) * 100, 1)
    """, (min_score, score_range))
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all_buzz(conn):
    """Fetch buzz data for all movies that haven't been fetched yet."""
    ensure_buzz_table(conn)
    unfetched = get_unfetched_movies(conn)

    if not unfetched:
        print("  All movies already have buzz data. Use --force to re-fetch.")
        return

    total = len(unfetched)
    print(f"  Fetching buzz data for {total} movies...")

    for i, (movie_id, title) in enumerate(unfetched):
        query = f'"{title}" 2026 box office prediction'
        snippets = search_web(query)
        signals = compute_buzz_signals(snippets)
        raw_score = compute_raw_buzz_score(signals)
        store_buzz(conn, movie_id, raw_score, signals)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            conn.commit()
            print(f"    Fetched: {i + 1}/{total} — {title}: {raw_score:.1f}")

        time.sleep(REQUEST_DELAY)

    conn.commit()

    # Normalize scores to 0-100
    print("  Normalizing buzz scores...")
    normalize_buzz_scores(conn)
    print("  Done.")


def retry_zero_buzz(conn):
    """Re-fetch buzz data for movies that got 0 sources (likely due to rate limiting)."""
    ensure_buzz_table(conn)
    zeros = get_zero_source_movies(conn)

    if not zeros:
        print("  No zero-source movies to retry.")
        return

    total = len(zeros)
    print(f"  Retrying buzz fetch for {total} movies with 0 sources (delay={RETRY_DELAY}s)...")

    updated = 0
    for i, (movie_id, title) in enumerate(zeros):
        query = f'"{title}" 2026 box office prediction'
        snippets = search_web(query)
        if not snippets:
            # Still no results — skip but don't overwrite
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"    Progress: {i + 1}/{total} — {title}: still 0")
            time.sleep(RETRY_DELAY)
            continue

        signals = compute_buzz_signals(snippets)
        raw_score = compute_raw_buzz_score(signals)
        store_buzz(conn, movie_id, raw_score, signals)
        updated += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            conn.commit()
            print(f"    Progress: {i + 1}/{total} — {title}: {raw_score:.1f}")

        time.sleep(RETRY_DELAY)

    conn.commit()
    print(f"  Updated {updated}/{total} movies.")

    # Re-normalize
    print("  Re-normalizing buzz scores...")
    normalize_buzz_scores(conn)
    print("  Done.")


# ---------------------------------------------------------------------------
# Backtest buzz: fetch buzz for historical movies in backtest.db
# ---------------------------------------------------------------------------

def ensure_backtest_buzz_table(conn):
    """Create backtest_buzz table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_buzz (
            tmdb_id     INTEGER NOT NULL,
            year        INTEGER NOT NULL,
            buzz_score  REAL DEFAULT 0,
            raw_data    TEXT,
            fetched_at  TEXT,
            UNIQUE(tmdb_id, year)
        )
    """)
    conn.commit()


def get_unfetched_backtest_movies(conn):
    """Return list of (tmdb_id, year, title) for backtest movies not yet fetched."""
    cursor = conn.execute("""
        SELECT bm.tmdb_id, bm.year, bm.title
        FROM backtest_movies bm
        LEFT JOIN backtest_buzz bb ON bb.tmdb_id = bm.tmdb_id AND bb.year = bm.year
        WHERE bb.tmdb_id IS NULL
        ORDER BY bm.year, bm.actual_rank
    """)
    return cursor.fetchall()


def fetch_backtest_buzz(conn, force=False):
    """Fetch buzz data for all backtest movies."""
    ensure_backtest_buzz_table(conn)

    if force:
        conn.execute("DELETE FROM backtest_buzz")
        conn.commit()

    unfetched = get_unfetched_backtest_movies(conn)
    if not unfetched:
        print("  All backtest movies already have buzz data. Use --force to re-fetch.")
        return

    total = len(unfetched)
    print(f"  Fetching buzz data for {total} backtest movies...")

    for i, (tmdb_id, year, title) in enumerate(unfetched):
        query = f'"{title}" {year} box office'
        snippets = search_web(query)
        signals = compute_buzz_signals(snippets)
        raw_score = compute_raw_buzz_score(signals)

        conn.execute("""
            INSERT OR REPLACE INTO backtest_buzz (tmdb_id, year, buzz_score, raw_data, fetched_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            tmdb_id, year, raw_score, json.dumps(signals),
            datetime.now(timezone.utc).isoformat(),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == total:
            conn.commit()
            print(f"    Fetched: {i + 1}/{total} — [{year}] {title}: {raw_score:.1f}")

        time.sleep(REQUEST_DELAY)

    conn.commit()

    # Normalize buzz scores to 0-100 across all backtest movies
    print("  Normalizing backtest buzz scores...")
    cursor = conn.execute("SELECT MAX(buzz_score), MIN(buzz_score) FROM backtest_buzz")
    max_score, min_score = cursor.fetchone()

    if max_score is not None and max_score != min_score:
        score_range = max_score - min_score
        conn.execute("""
            UPDATE backtest_buzz
            SET buzz_score = ROUND(((buzz_score - ?) / ?) * 100, 1)
        """, (min_score, score_range))
    else:
        conn.execute("UPDATE backtest_buzz SET buzz_score = 0")
    conn.commit()
    print("  Done.")


def main():
    import sys

    mode = "backtest" if "--backtest" in sys.argv else "2026"
    force = "--force" in sys.argv
    retry_zeros = "--retry-zeros" in sys.argv

    if mode == "backtest":
        print("=" * 64)
        print("  BUZZ DATA FETCHER — BACKTEST MOVIES (2015-2025)")
        print("=" * 64)

        conn = sqlite3.connect(BACKTEST_DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        fetch_backtest_buzz(conn, force=force)

        # Print top 10 by buzz per sample years
        for year in [2019, 2022, 2025]:
            print(f"\n  Top 5 by buzz ({year}):")
            cursor = conn.execute("""
                SELECT bm.title, bb.buzz_score, bb.raw_data
                FROM backtest_buzz bb
                JOIN backtest_movies bm ON bm.tmdb_id = bb.tmdb_id AND bm.year = bb.year
                WHERE bb.year = ?
                ORDER BY bb.buzz_score DESC
                LIMIT 5
            """, (year,))
            for rank, (title, score, raw) in enumerate(cursor.fetchall(), 1):
                data = json.loads(raw) if raw else {}
                gross = data.get("predicted_gross", 0)
                print(f"    {rank}. {title:<40} buzz={score:>5.1f}  gross=${gross:.0f}M")

        conn.close()
    else:
        print("=" * 64)
        print("  BUZZ DATA FETCHER — 2026 BOX OFFICE PREDICTIONS")
        print("=" * 64)

        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")

        if force:
            print("  Force mode: clearing existing buzz data...")
            conn.execute("DELETE FROM movie_buzz")
            conn.commit()

        if retry_zeros:
            retry_zero_buzz(conn)
        else:
            fetch_all_buzz(conn)

        # Print top 10 by buzz
        print("\n  Top 10 by buzz score:")
        cursor = conn.execute("""
            SELECT m.title, mb.buzz_score, mb.raw_data
            FROM movie_buzz mb
            JOIN movies m ON m.id = mb.movie_id
            ORDER BY mb.buzz_score DESC
            LIMIT 10
        """)
        for rank, (title, score, raw) in enumerate(cursor.fetchall(), 1):
            data = json.loads(raw) if raw else {}
            gross = data.get("predicted_gross", 0)
            print(f"    {rank:>2}. {title:<40} buzz={score:>5.1f}  gross=${gross:.0f}M")

        conn.close()

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
