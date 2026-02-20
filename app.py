"""Flask app serving the Movies Box Office Pool Draft Tool."""

import os
import json
import sqlite3
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "movies.db")
BACKTEST_DB_PATH = os.path.join(os.path.dirname(__file__), "backtest.db")


def get_db():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_backtest_db():
    """Get a backtest database connection with row factory."""
    conn = sqlite3.connect(BACKTEST_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/movies")
def api_movies():
    """Return all movies with scores. Supports filtering and sorting."""
    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT
            m.id, m.tmdb_id, m.title, m.release_date, m.budget, m.genre,
            m.overview, m.poster_path, m.popularity, m.belongs_to_collection,
            ms.director_score, ms.producer_score, ms.cast_score, ms.buzz_score, ms.total_score
        FROM movies m
        LEFT JOIN movie_scores ms ON ms.movie_id = m.id
        WHERE 1=1
    """
    params = []

    # Genre filter
    genre = request.args.get("genre")
    if genre:
        query += " AND m.genre LIKE ?"
        params.append(f"%{genre}%")

    # Quarter filter
    quarter = request.args.get("quarter")
    if quarter:
        q_map = {"Q1": ("01", "03"), "Q2": ("04", "06"), "Q3": ("07", "09"), "Q4": ("10", "12")}
        if quarter in q_map:
            start_m, end_m = q_map[quarter]
            query += " AND substr(m.release_date, 6, 2) BETWEEN ? AND ?"
            params.extend([start_m, end_m])

    # Minimum score filter
    min_score = request.args.get("min_score")
    if min_score:
        query += " AND ms.total_score >= ?"
        params.append(float(min_score))

    # Search
    search = request.args.get("search")
    if search:
        query += " AND (m.title LIKE ? OR m.id IN (SELECT mc.movie_id FROM movie_credits mc JOIN people p ON p.id = mc.person_id WHERE p.name LIKE ?))"
        params.extend([f"%{search}%", f"%{search}%"])

    # Sorting
    sort = request.args.get("sort", "total_score")
    order = request.args.get("order", "desc")
    allowed_sorts = {
        "total_score": "ms.total_score",
        "title": "m.title",
        "release_date": "m.release_date",
        "director_score": "ms.director_score",
        "cast_score": "ms.cast_score",
        "producer_score": "ms.producer_score",
        "buzz_score": "ms.buzz_score",
        "popularity": "m.popularity",
    }
    sort_col = allowed_sorts.get(sort, "ms.total_score")
    order_dir = "ASC" if order == "asc" else "DESC"
    query += f" ORDER BY {sort_col} {order_dir} NULLS LAST"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Also fetch director and top cast names for each movie
    movies = []
    for row in rows:
        movie = dict(row)

        # Get director name(s)
        cursor.execute("""
            SELECT p.name FROM movie_credits mc
            JOIN people p ON p.id = mc.person_id
            WHERE mc.movie_id = ? AND mc.role = 'director'
        """, (movie["id"],))
        movie["directors"] = [r["name"] for r in cursor.fetchall()]

        # Get top cast names
        cursor.execute("""
            SELECT p.name FROM movie_credits mc
            JOIN people p ON p.id = mc.person_id
            WHERE mc.movie_id = ? AND mc.role = 'cast'
            ORDER BY mc.billing_order
            LIMIT 3
        """, (movie["id"],))
        movie["top_cast"] = [r["name"] for r in cursor.fetchall()]

        movies.append(movie)

    conn.close()
    return jsonify(movies)


@app.route("/api/movie/<int:movie_id>")
def api_movie_detail(movie_id):
    """Return detailed movie info with full credit breakdown and track records."""
    conn = get_db()
    cursor = conn.cursor()

    # Movie info
    cursor.execute("SELECT * FROM movies WHERE id = ?", (movie_id,))
    movie = cursor.fetchone()
    if not movie:
        conn.close()
        return jsonify({"error": "Movie not found"}), 404

    result = dict(movie)

    # Score breakdown
    cursor.execute("SELECT * FROM movie_scores WHERE movie_id = ?", (movie_id,))
    score_row = cursor.fetchone()
    if score_row:
        result["scores"] = dict(score_row)
        if score_row["score_breakdown"]:
            result["score_breakdown"] = json.loads(score_row["score_breakdown"])
    else:
        result["scores"] = None
        result["score_breakdown"] = None

    # Buzz data
    cursor.execute("SELECT buzz_score, raw_data FROM movie_buzz WHERE movie_id = ?", (movie_id,))
    buzz_row = cursor.fetchone()
    if buzz_row:
        result["buzz"] = {
            "buzz_score": buzz_row["buzz_score"],
            "raw_data": json.loads(buzz_row["raw_data"]) if buzz_row["raw_data"] else None,
        }
    else:
        result["buzz"] = None

    # Credits with person history
    cursor.execute("""
        SELECT mc.role, mc.billing_order, mc.character_name,
               p.id as person_id, p.tmdb_id as person_tmdb_id, p.name, p.profile_path
        FROM movie_credits mc
        JOIN people p ON p.id = mc.person_id
        WHERE mc.movie_id = ?
        ORDER BY mc.role, mc.billing_order
    """, (movie_id,))
    credits = cursor.fetchall()

    result["credits"] = []
    for credit in credits:
        c = dict(credit)
        # Fetch person history
        cursor.execute("""
            SELECT movie_title, release_date, budget, revenue, profit
            FROM person_history
            WHERE person_id = ?
            ORDER BY release_date DESC
        """, (c["person_id"],))
        c["history"] = [dict(h) for h in cursor.fetchall()]
        result["credits"].append(c)

    conn.close()
    return jsonify(result)


@app.route("/api/person/<int:person_id>")
def api_person_detail(person_id):
    """Return a person's track record details."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM people WHERE id = ?", (person_id,))
    person = cursor.fetchone()
    if not person:
        conn.close()
        return jsonify({"error": "Person not found"}), 404

    result = dict(person)

    # History
    cursor.execute("""
        SELECT movie_title, release_date, budget, revenue, profit
        FROM person_history
        WHERE person_id = ?
        ORDER BY release_date DESC
    """, (person_id,))
    result["history"] = [dict(h) for h in cursor.fetchall()]

    # Movies they're associated with in 2026
    cursor.execute("""
        SELECT m.id, m.title, m.release_date, mc.role, mc.character_name
        FROM movie_credits mc
        JOIN movies m ON m.id = mc.movie_id
        WHERE mc.person_id = ?
    """, (person_id,))
    result["upcoming_movies"] = [dict(r) for r in cursor.fetchall()]

    conn.close()
    return jsonify(result)


@app.route("/api/genres")
def api_genres():
    """Return distinct genre list for filter dropdown."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT genre FROM movies WHERE genre != '' AND genre IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    # Split comma-separated genres and deduplicate
    genres = set()
    for row in rows:
        for g in row["genre"].split(", "):
            g = g.strip()
            if g:
                genres.add(g)

    return jsonify(sorted(genres))


@app.route("/api/backtest/summary")
def api_backtest_summary():
    """Return backtest summary across all years."""
    if not os.path.exists(BACKTEST_DB_PATH):
        return jsonify({"error": "Backtest data not available. Run backtest.py first."}), 404
    conn = get_backtest_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM backtest_results ORDER BY year")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/backtest/year/<int:year>")
def api_backtest_year(year):
    """Return per-movie backtest data for a specific year."""
    if not os.path.exists(BACKTEST_DB_PATH):
        return jsonify({"error": "Backtest data not available. Run backtest.py first."}), 404
    conn = get_backtest_db()
    cursor = conn.cursor()

    # Year summary
    cursor.execute("SELECT * FROM backtest_results WHERE year = ?", (year,))
    summary = cursor.fetchone()
    if not summary:
        conn.close()
        return jsonify({"error": f"No backtest data for {year}"}), 404

    # Movies for this year
    cursor.execute(
        "SELECT * FROM backtest_movies WHERE year = ? ORDER BY predicted_rank",
        (year,)
    )
    movies = cursor.fetchall()
    conn.close()

    return jsonify({
        "summary": dict(summary),
        "movies": [dict(m) for m in movies],
    })


if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        print("Run 'python fetch_data.py' first to populate the database.")
        exit(1)

    app.run(debug=True, port=5050)
