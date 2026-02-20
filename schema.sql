CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER UNIQUE NOT NULL,
    title TEXT NOT NULL,
    release_date TEXT,
    budget INTEGER DEFAULT 0,
    genre TEXT,
    overview TEXT,
    poster_path TEXT,
    popularity REAL DEFAULT 0,
    belongs_to_collection TEXT
);

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    profile_path TEXT
);

CREATE TABLE IF NOT EXISTS movie_credits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id INTEGER NOT NULL,
    person_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('director', 'producer', 'cast')),
    billing_order INTEGER,
    character_name TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies(id),
    FOREIGN KEY (person_id) REFERENCES people(id),
    UNIQUE(movie_id, person_id, role)
);

CREATE TABLE IF NOT EXISTS person_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    tmdb_movie_id INTEGER NOT NULL,
    movie_title TEXT,
    release_date TEXT,
    budget INTEGER DEFAULT 0,
    revenue INTEGER DEFAULT 0,
    profit INTEGER DEFAULT 0,
    FOREIGN KEY (person_id) REFERENCES people(id),
    UNIQUE(person_id, tmdb_movie_id)
);

CREATE TABLE IF NOT EXISTS movie_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id INTEGER UNIQUE NOT NULL,
    director_score REAL DEFAULT 0,
    producer_score REAL DEFAULT 0,
    cast_score REAL DEFAULT 0,
    buzz_score REAL DEFAULT 0,
    total_score REAL DEFAULT 0,
    score_breakdown TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies(id)
);

CREATE INDEX IF NOT EXISTS idx_movie_credits_movie ON movie_credits(movie_id);
CREATE INDEX IF NOT EXISTS idx_movie_credits_person ON movie_credits(person_id);
CREATE INDEX IF NOT EXISTS idx_person_history_person ON person_history(person_id);
CREATE TABLE IF NOT EXISTS movie_buzz (
    movie_id    INTEGER UNIQUE NOT NULL,
    buzz_score  REAL DEFAULT 0,
    raw_data    TEXT,
    fetched_at  TEXT,
    FOREIGN KEY (movie_id) REFERENCES movies(id)
);

CREATE INDEX IF NOT EXISTS idx_movie_scores_total ON movie_scores(total_score DESC);
