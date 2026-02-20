"""Scoring algorithm for the box office draft pool."""

import json
import sqlite3


# Recency weights for last 3 movies (most recent first)
RECENCY_WEIGHTS = [0.6, 0.2, 0.2]

# Consistency bonuses based on how many of last 3 movies were profitable
CONSISTENCY_MAP = {
    3: 0.6,
    2: 0.6,
    1: 0.6,
    0: 0.2,
}

# Movie score component weights
WEIGHT_DIRECTOR = 0.2
WEIGHT_PRODUCER = 0.2
WEIGHT_CAST = 0.05
WEIGHT_FRANCHISE = 0.5
WEIGHT_BUZZ = 0.05


def compute_person_score(history_rows):
    """
    Compute a talent score for a person based on their last 3 movies.

    history_rows: list of (movie_title, release_date, budget, revenue, profit)
                  sorted by release_date descending (most recent first).

    Returns (score, breakdown_list).
    """
    if not history_rows:
        return 0.0, []

    # Cap to 3 most recent
    rows = history_rows[:3]

    profits = []
    breakdown = []
    profitable_count = 0

    for i, row in enumerate(rows):
        title, rel_date, budget, revenue, profit = row
        weight = RECENCY_WEIGHTS[i] if i < len(RECENCY_WEIGHTS) else 0.3

        # Use profit in millions for readability
        profit_m = profit / 1_000_000 if profit else 0
        profits.append(profit_m * weight)

        if profit > 0:
            profitable_count += 1

        breakdown.append({
            "title": title,
            "release_date": rel_date,
            "budget": budget,
            "revenue": revenue,
            "profit": profit,
            "weight": weight,
            "weighted_profit_m": round(profit_m * weight, 2),
        })

    consistency = CONSISTENCY_MAP.get(profitable_count, 0.5)
    raw_score = sum(profits) * consistency

    return raw_score, breakdown


def get_buzz_score(movie_db_id, conn):
    """Read buzz score from movie_buzz table. Returns 0.0 if not found."""
    row = conn.execute(
        "SELECT buzz_score FROM movie_buzz WHERE movie_id = ?",
        (movie_db_id,),
    ).fetchone()
    return row[0] if row else 0.0


def compute_franchise_bonus(movie_row, conn):
    """
    Compute a franchise bonus if the movie belongs to a collection.
    Returns a score based on the collection's average historical profit.
    """
    collection_json = movie_row.get("belongs_to_collection")
    if not collection_json:
        return 0.0, None

    try:
        collection = json.loads(collection_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0, None

    if not collection or not collection.get("id"):
        return 0.0, None

    # Franchise movies tend to do well — assign a flat bonus scaled by
    # the movie's own popularity as a proxy (detailed collection revenue
    # would require extra API calls).
    # A known franchise gets a base bonus of 176 (in score units).
    bonus = 176.0
    return bonus, {"collection_name": collection.get("name", "Unknown"), "bonus": bonus}


def compute_movie_score(movie_db_id, conn):
    """Compute the full draft score for a single movie."""
    cursor = conn.cursor()

    # Get movie info
    cursor.execute("SELECT * FROM movies WHERE id = ?", (movie_db_id,))
    movie = cursor.fetchone()
    if not movie:
        return None

    col_names = [desc[0] for desc in cursor.description]
    movie_dict = dict(zip(col_names, movie))

    # Get credits grouped by role
    cursor.execute("""
        SELECT mc.role, mc.billing_order, p.id as person_id, p.name
        FROM movie_credits mc
        JOIN people p ON p.id = mc.person_id
        WHERE mc.movie_id = ?
        ORDER BY mc.role, mc.billing_order
    """, (movie_db_id,))
    credits = cursor.fetchall()

    director_scores = []
    producer_scores = []
    cast_scores = []
    full_breakdown = {"directors": [], "producers": [], "cast": []}

    for role, billing_order, person_id, person_name in credits:
        # Get person's history
        cursor.execute("""
            SELECT movie_title, release_date, budget, revenue, profit
            FROM person_history
            WHERE person_id = ?
            ORDER BY release_date DESC
        """, (person_id,))
        history = cursor.fetchall()

        score, breakdown = compute_person_score(history)
        entry = {"name": person_name, "person_id": person_id, "score": round(score, 2), "history": breakdown}

        if role == "director":
            director_scores.append(score)
            full_breakdown["directors"].append(entry)
        elif role == "producer":
            producer_scores.append(score)
            full_breakdown["producers"].append(entry)
        elif role == "cast":
            cast_scores.append(score)
            full_breakdown["cast"].append(entry)

    # Aggregate scores per role
    dir_score = max(director_scores) if director_scores else 0
    prod_score = max(producer_scores) if producer_scores else 0
    cast_avg = sum(cast_scores[:3]) / len(cast_scores[:3]) if cast_scores else 0

    # Franchise bonus
    franchise_score, franchise_info = compute_franchise_bonus(movie_dict, conn)
    if franchise_info:
        full_breakdown["franchise"] = franchise_info

    # Buzz score
    buzz = get_buzz_score(movie_db_id, conn)
    full_breakdown["buzz"] = {"buzz_score": buzz}

    # Weighted total
    total = (
        dir_score * WEIGHT_DIRECTOR +
        prod_score * WEIGHT_PRODUCER +
        cast_avg * WEIGHT_CAST +
        franchise_score * WEIGHT_FRANCHISE +
        buzz * WEIGHT_BUZZ
    )

    return {
        "movie_id": movie_db_id,
        "director_score": round(dir_score, 2),
        "producer_score": round(prod_score, 2),
        "cast_score": round(cast_avg, 2),
        "buzz_score": round(buzz, 2),
        "total_score": round(total, 2),
        "breakdown": full_breakdown,
    }


def normalize_scores(conn):
    """Normalize all movie scores to a 0-100 scale."""
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(total_score), MIN(total_score) FROM movie_scores")
    row = cursor.fetchone()
    max_score, min_score = row

    if max_score is None or max_score == min_score:
        # Can't normalize, set all to 50
        cursor.execute("UPDATE movie_scores SET total_score = 50")
        conn.commit()
        return

    score_range = max_score - min_score
    cursor.execute("SELECT id, total_score, director_score, producer_score, cast_score, buzz_score FROM movie_scores")
    rows = cursor.fetchall()

    for row_id, total, dir_s, prod_s, cast_s, buzz_s in rows:
        normalized_total = ((total - min_score) / score_range) * 100
        # Normalize component scores proportionally
        if total != 0:
            ratio = normalized_total / total if total != 0 else 0
        else:
            ratio = 0
        cursor.execute("""
            UPDATE movie_scores
            SET total_score = ?, director_score = ?, producer_score = ?, cast_score = ?, buzz_score = ?
            WHERE id = ?
        """, (
            round(normalized_total, 1),
            round(dir_s * ratio, 1) if ratio else 0,
            round(prod_s * ratio, 1) if ratio else 0,
            round(cast_s * ratio, 1) if ratio else 0,
            round(buzz_s * ratio, 1) if ratio else 0,
            row_id,
        ))

    conn.commit()


def _ensure_buzz_column(conn):
    """Add buzz_score column to movie_scores if it doesn't exist."""
    cursor = conn.execute("PRAGMA table_info(movie_scores)")
    columns = {row[1] for row in cursor.fetchall()}
    if "buzz_score" not in columns:
        conn.execute("ALTER TABLE movie_scores ADD COLUMN buzz_score REAL DEFAULT 0")
        conn.commit()


def compute_all_scores(conn):
    """Compute and store scores for all movies."""
    _ensure_buzz_column(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM movies")
    movie_ids = [row[0] for row in cursor.fetchall()]

    print(f"  Computing scores for {len(movie_ids)} movies...")

    for i, movie_id in enumerate(movie_ids):
        result = compute_movie_score(movie_id, conn)
        if not result:
            continue

        cursor.execute("""
            INSERT OR REPLACE INTO movie_scores (movie_id, director_score, producer_score, cast_score, buzz_score, total_score, score_breakdown)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["movie_id"],
            result["director_score"],
            result["producer_score"],
            result["cast_score"],
            result["buzz_score"],
            result["total_score"],
            json.dumps(result["breakdown"]),
        ))

        if (i + 1) % 50 == 0:
            print(f"    Scored: {i + 1}/{len(movie_ids)}")
            conn.commit()

    conn.commit()

    # Normalize to 0-100
    normalize_scores(conn)
    print("  Scores computed and normalized.")
