"""Fetch 2026 movie data from TMDB and populate the SQLite database."""

import os
import sys
import json
import time
import sqlite3
import requests
from dotenv import load_dotenv
from scoring import compute_all_scores

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
DB_PATH = os.path.join(os.path.dirname(__file__), "movies.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

# Rate limiting: small delay between requests
REQUEST_DELAY = 0.05  # 50ms between requests (~20 req/sec, well under 40 limit)


def api_get(endpoint, params=None):
    """Make a GET request to TMDB API with rate limiting."""
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    time.sleep(REQUEST_DELAY)
    resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def init_db():
    """Initialize the database with schema."""
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def fetch_2026_movies(conn):
    """Fetch all 2026 movie releases from TMDB discover endpoint."""
    print("Fetching 2026 movies...")
    all_movies = []
    page = 1

    while True:
        data = api_get("/discover/movie", {
            "primary_release_date.gte": "2026-01-01",
            "primary_release_date.lte": "2026-12-31",
            "sort_by": "popularity.desc",
            "page": page,
        })

        results = data.get("results", [])
        if not results:
            break

        all_movies.extend(results)
        total_pages = data.get("total_pages", 1)
        print(f"  Page {page}/{total_pages} — {len(all_movies)} movies so far")

        if page >= total_pages:
            break
        page += 1

    # Filter: skip very low popularity movies (likely no real release)
    filtered = [m for m in all_movies if m.get("popularity", 0) >= 3.0]
    print(f"  Filtered to {len(filtered)} movies (popularity >= 3.0)")

    # Insert into DB
    cursor = conn.cursor()
    for m in filtered:
        genres = ", ".join(g["name"] for g in m.get("genres", []))
        # Genre IDs from discover don't include names; we'll fetch details later if needed
        cursor.execute("""
            INSERT OR IGNORE INTO movies (tmdb_id, title, release_date, budget, genre, overview, poster_path, popularity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            m["id"],
            m["title"],
            m.get("release_date", ""),
            m.get("budget", 0),
            genres,
            m.get("overview", ""),
            m.get("poster_path", ""),
            m.get("popularity", 0),
        ))
    conn.commit()

    # Now fetch detailed info for each movie (budget, genres, collection)
    cursor.execute("SELECT id, tmdb_id FROM movies")
    db_movies = cursor.fetchall()
    print(f"\nFetching details for {len(db_movies)} movies...")

    for i, (db_id, tmdb_id) in enumerate(db_movies):
        try:
            details = api_get(f"/movie/{tmdb_id}")
            genre_names = ", ".join(g["name"] for g in details.get("genres", []))
            collection = json.dumps(details["belongs_to_collection"]) if details.get("belongs_to_collection") else None
            cursor.execute("""
                UPDATE movies SET budget = ?, genre = ?, belongs_to_collection = ?
                WHERE id = ?
            """, (details.get("budget", 0), genre_names, collection, db_id))
        except Exception as e:
            print(f"  Warning: Could not fetch details for movie {tmdb_id}: {e}")

        if (i + 1) % 25 == 0:
            print(f"  Details: {i + 1}/{len(db_movies)}")
            conn.commit()

    conn.commit()
    print(f"  Done fetching movie details.")
    return len(db_movies)


def fetch_credits(conn):
    """Fetch credits (director, producers, top cast) for each movie."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, tmdb_id FROM movies")
    movies = cursor.fetchall()

    print(f"\nFetching credits for {len(movies)} movies...")

    for i, (movie_db_id, tmdb_id) in enumerate(movies):
        try:
            data = api_get(f"/movie/{tmdb_id}/credits")
        except Exception as e:
            print(f"  Warning: Could not fetch credits for movie {tmdb_id}: {e}")
            continue

        cast = data.get("cast", [])
        crew = data.get("crew", [])

        # Director(s)
        directors = [c for c in crew if c.get("job") == "Director"]
        for d in directors:
            person_id = _upsert_person(cursor, d["id"], d["name"], d.get("profile_path"))
            cursor.execute("""
                INSERT OR IGNORE INTO movie_credits (movie_id, person_id, role, billing_order)
                VALUES (?, ?, 'director', 0)
            """, (movie_db_id, person_id))

        # Executive Producer(s) — take top 2
        producers = [c for c in crew if c.get("job") in ("Producer", "Executive Producer")]
        seen_producers = set()
        for p in producers[:3]:
            if p["id"] in seen_producers:
                continue
            seen_producers.add(p["id"])
            person_id = _upsert_person(cursor, p["id"], p["name"], p.get("profile_path"))
            cursor.execute("""
                INSERT OR IGNORE INTO movie_credits (movie_id, person_id, role, billing_order)
                VALUES (?, ?, 'producer', ?)
            """, (movie_db_id, person_id, producers.index(p)))

        # Top 5 billed cast
        for order, c in enumerate(cast[:5]):
            person_id = _upsert_person(cursor, c["id"], c["name"], c.get("profile_path"))
            cursor.execute("""
                INSERT OR IGNORE INTO movie_credits (movie_id, person_id, role, billing_order, character_name)
                VALUES (?, ?, 'cast', ?, ?)
            """, (movie_db_id, person_id, order, c.get("character", "")))

        if (i + 1) % 25 == 0:
            print(f"  Credits: {i + 1}/{len(movies)}")
            conn.commit()

    conn.commit()
    print("  Done fetching credits.")


def _upsert_person(cursor, tmdb_id, name, profile_path):
    """Insert a person if not exists, return their DB id."""
    cursor.execute("SELECT id FROM people WHERE tmdb_id = ?", (tmdb_id,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute("""
        INSERT INTO people (tmdb_id, name, profile_path)
        VALUES (?, ?, ?)
    """, (tmdb_id, name, profile_path))
    return cursor.lastrowid


def fetch_person_histories(conn):
    """Fetch the last 3 released movies for each person to build track records."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, tmdb_id, name FROM people")
    people = cursor.fetchall()

    print(f"\nFetching track records for {len(people)} people...")

    # Cache movie details to avoid re-fetching
    movie_detail_cache = {}

    for i, (person_db_id, person_tmdb_id, person_name) in enumerate(people):
        # Check if we already have history for this person
        cursor.execute("SELECT COUNT(*) FROM person_history WHERE person_id = ?", (person_db_id,))
        if cursor.fetchone()[0] > 0:
            continue

        try:
            data = api_get(f"/person/{person_tmdb_id}/movie_credits")
        except Exception as e:
            print(f"  Warning: Could not fetch filmography for {person_name}: {e}")
            continue

        # Combine cast and crew appearances
        all_movies = []
        for m in data.get("cast", []):
            if m.get("release_date"):
                all_movies.append(m)
        for m in data.get("crew", []):
            if m.get("release_date") and m.get("job") in ("Director", "Producer", "Executive Producer"):
                all_movies.append(m)

        # Deduplicate by tmdb movie id
        seen = set()
        unique_movies = []
        for m in all_movies:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique_movies.append(m)

        # Sort by release date descending, take only released movies (before 2026)
        released = [m for m in unique_movies if m.get("release_date", "") < "2026-01-01" and m.get("release_date", "") > "2000-01-01"]
        released.sort(key=lambda x: x.get("release_date", ""), reverse=True)

        # Take the 3 most recent
        recent = released[:3]

        for m in recent:
            movie_tmdb_id = m["id"]

            # Fetch detailed movie info (budget/revenue)
            if movie_tmdb_id not in movie_detail_cache:
                try:
                    details = api_get(f"/movie/{movie_tmdb_id}")
                    movie_detail_cache[movie_tmdb_id] = details
                except Exception:
                    movie_detail_cache[movie_tmdb_id] = None

            details = movie_detail_cache.get(movie_tmdb_id)
            if not details:
                continue

            budget = details.get("budget", 0)
            revenue = details.get("revenue", 0)
            profit = revenue - budget if (budget > 0 and revenue > 0) else 0

            cursor.execute("""
                INSERT OR IGNORE INTO person_history (person_id, tmdb_movie_id, movie_title, release_date, budget, revenue, profit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                person_db_id,
                movie_tmdb_id,
                details.get("title", m.get("title", "")),
                details.get("release_date", ""),
                budget,
                revenue,
                profit,
            ))

        if (i + 1) % 50 == 0:
            print(f"  Histories: {i + 1}/{len(people)}")
            conn.commit()

    conn.commit()
    print("  Done fetching person histories.")


def main():
    if not API_KEY:
        print("Error: TMDB_API_KEY not set. Create a .env file with your API key.")
        print("Get a free key at https://www.themoviedb.org/settings/api")
        sys.exit(1)

    print("=== Movies Box Office Pool Draft Tool — Data Fetch ===\n")

    conn = init_db()
    try:
        num_movies = fetch_2026_movies(conn)
        fetch_credits(conn)
        fetch_person_histories(conn)

        print("\n--- Computing scores ---")
        compute_all_scores(conn)

        # Summary
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM movies")
        m_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM people")
        p_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM person_history")
        h_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM movie_scores")
        s_count = cursor.fetchone()[0]

        print(f"\n=== Done! ===")
        print(f"  Movies:    {m_count}")
        print(f"  People:    {p_count}")
        print(f"  Histories: {h_count}")
        print(f"  Scores:    {s_count}")
        print(f"\nDatabase saved to: {DB_PATH}")
        print("Run 'python app.py' to start the web server.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
