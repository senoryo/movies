"""Optimize scoring weights via grid search on 10 years of backtest data.

Performs coordinate descent over 7 parameter groups, using pre-computed layers
to avoid redundant work. Updates scoring.py constants in-place when done.

Usage:
    python optimize.py
"""

import json
import os
import re
import sqlite3
from itertools import product

DB_PATH = os.path.join(os.path.dirname(__file__), "backtest.db")
SCORING_PATH = os.path.join(os.path.dirname(__file__), "scoring.py")
YEARS = list(range(2021, 2026))


# ---------------------------------------------------------------------------
# Layer 0: Load raw data from cache (once)
# ---------------------------------------------------------------------------

def load_all_data(conn):
    """Load movie_cache, credits_cache, person_filmography_cache into dicts."""
    # Movie lookup: tmdb_id -> {budget, revenue, release_date, collection}
    movie_lookup = {}
    for row in conn.execute(
        "SELECT tmdb_id, title, release_date, budget, revenue, belongs_to_collection "
        "FROM movie_cache"
    ):
        tmdb_id, title, release_date, budget, revenue, collection = row
        movie_lookup[tmdb_id] = {
            "title": title,
            "release_date": release_date or "",
            "budget": budget or 0,
            "revenue": revenue or 0,
            "collection": collection,
        }

    # Filmography lookup: person_id -> sorted list of {id, release_date}
    filmography_lookup = {}
    for row in conn.execute(
        "SELECT tmdb_person_id, filmography_json FROM person_filmography_cache"
    ):
        person_id, filmography_json = row
        films = json.loads(filmography_json)
        # Sort by release_date descending for quick slicing later
        films.sort(key=lambda x: x.get("release_date", ""), reverse=True)
        filmography_lookup[person_id] = films

    # Credits lookup: tmdb_movie_id -> {directors: [{id, name}], producers, cast}
    credits_lookup = {}
    for row in conn.execute(
        "SELECT tmdb_movie_id, credits_json FROM credits_cache"
    ):
        tmdb_movie_id, credits_json = row
        credits_lookup[tmdb_movie_id] = json.loads(credits_json)

    # Backtest movies: year -> list of {tmdb_id, title, revenue, budget, actual_rank}
    backtest_movies_by_year = {}
    for row in conn.execute(
        "SELECT year, tmdb_id, title, budget, revenue, actual_rank "
        "FROM backtest_movies ORDER BY year, actual_rank"
    ):
        year, tmdb_id, title, budget, revenue, actual_rank = row
        backtest_movies_by_year.setdefault(year, []).append({
            "tmdb_id": tmdb_id,
            "title": title,
            "budget": budget,
            "revenue": revenue,
            "actual_rank": actual_rank,
        })

    # Buzz lookup: (tmdb_id, year) -> buzz_score
    buzz_lookup = {}
    try:
        for row in conn.execute(
            "SELECT tmdb_id, year, buzz_score FROM backtest_buzz"
        ):
            tmdb_id, year, buzz_score = row
            buzz_lookup[(tmdb_id, year)] = buzz_score or 0.0
    except Exception:
        pass  # Table may not exist yet

    return movie_lookup, filmography_lookup, credits_lookup, backtest_movies_by_year, buzz_lookup


# ---------------------------------------------------------------------------
# Layer 1: Pre-compute per-(person, year) recent movie tuples & movie credits
# ---------------------------------------------------------------------------

def precompute_layer1(movie_lookup, filmography_lookup, credits_lookup,
                      backtest_movies_by_year, buzz_lookup=None):
    """
    Build:
      person_history[(person_id, year)] -> list of (profit_M, is_profitable) for up to 3 recent movies
      movie_credits_map[(tmdb_id, year)] -> {
          director_ids, producer_ids, cast_ids (top 3), is_franchise
      }
      actual_ranks_by_year[year] -> list of tmdb_ids sorted by actual revenue (rank 1 first)
    """
    # Collect all (person_id, year) pairs we need
    person_year_pairs = set()
    movie_credits_map = {}
    actual_ranks_by_year = {}

    for year, movies in backtest_movies_by_year.items():
        actual_ranks_by_year[year] = [m["tmdb_id"] for m in movies]

        for m in movies:
            tmdb_id = m["tmdb_id"]
            credits = credits_lookup.get(tmdb_id, {})
            director_ids = [d["id"] for d in credits.get("directors", [])]
            producer_ids = [p["id"] for p in credits.get("producers", [])]
            cast_ids = [c["id"] for c in credits.get("cast", [])[:5]]

            movie_info = movie_lookup.get(tmdb_id, {})
            is_franchise = bool(movie_info.get("collection"))

            movie_credits_map[(tmdb_id, year)] = {
                "director_ids": director_ids,
                "producer_ids": producer_ids,
                "cast_ids": cast_ids,
                "is_franchise": is_franchise,
            }

            for pid in director_ids + producer_ids + cast_ids:
                person_year_pairs.add((pid, year))

    # Pre-compute person history for each (person, year)
    person_history = {}
    for person_id, year in person_year_pairs:
        cutoff = f"{year}-01-01"
        films = filmography_lookup.get(person_id, [])

        # Filter: before cutoff, after 2000, with valid data
        recent = []
        for f in films:
            rd = f.get("release_date", "")
            if rd and rd < cutoff and rd > "2000-01-01":
                mid = f["id"]
                minfo = movie_lookup.get(mid)
                if minfo and minfo["budget"] > 0 and minfo["revenue"] > 0:
                    profit = minfo["revenue"] - minfo["budget"]
                    profit_m = profit / 1_000_000
                    recent.append((profit_m, profit > 0))
                    if len(recent) == 3:
                        break

        person_history[(person_id, year)] = recent

    return person_history, movie_credits_map, actual_ranks_by_year


# ---------------------------------------------------------------------------
# Layer 2: Compute person scores from history + recency/consistency params
# ---------------------------------------------------------------------------

def compute_person_scores(person_history, recency_weights, consistency_map):
    """
    person_scores[(person_id, year)] = float score
    """
    person_scores = {}
    for (person_id, year), history in person_history.items():
        if not history:
            person_scores[(person_id, year)] = 0.0
            continue

        weighted_profits = []
        profitable_count = 0
        for i, (profit_m, is_profitable) in enumerate(history):
            w = recency_weights[i] if i < len(recency_weights) else 0.3
            weighted_profits.append(profit_m * w)
            if is_profitable:
                profitable_count += 1

        consistency = consistency_map.get(profitable_count, 0.5)
        person_scores[(person_id, year)] = sum(weighted_profits) * consistency

    return person_scores


# ---------------------------------------------------------------------------
# Layer 3: Compute movie components from person scores + franchise bonus
# ---------------------------------------------------------------------------

def compute_movie_components(person_scores, movie_credits_map,
                             backtest_movies_by_year, franchise_bonus,
                             buzz_lookup=None):
    """
    movie_components[(tmdb_id, year)] = (dir_score, prod_score, cast_avg, fran_score, buzz_score)
    """
    if buzz_lookup is None:
        buzz_lookup = {}
    movie_components = {}
    for year, movies in backtest_movies_by_year.items():
        for m in movies:
            tmdb_id = m["tmdb_id"]
            creds = movie_credits_map.get((tmdb_id, year), {})

            # Director: max
            dir_scores = [
                person_scores.get((pid, year), 0.0)
                for pid in creds.get("director_ids", [])
            ]
            dir_score = max(dir_scores) if dir_scores else 0.0

            # Producer: max
            prod_scores = [
                person_scores.get((pid, year), 0.0)
                for pid in creds.get("producer_ids", [])
            ]
            prod_score = max(prod_scores) if prod_scores else 0.0

            # Cast: avg of top 3
            cast_ids = creds.get("cast_ids", [])
            cast_scores = [
                person_scores.get((pid, year), 0.0)
                for pid in cast_ids
            ]
            top3 = cast_scores[:3]
            cast_avg = sum(top3) / len(top3) if top3 else 0.0

            # Franchise
            fran_score = franchise_bonus if creds.get("is_franchise") else 0.0

            # Buzz: from backtest_buzz table if available
            buzz_score = buzz_lookup.get((tmdb_id, year), 0.0)

            movie_components[(tmdb_id, year)] = (dir_score, prod_score, cast_avg, fran_score, buzz_score)

    return movie_components


# ---------------------------------------------------------------------------
# Layer 4: Compute total scores, rank, evaluate
# ---------------------------------------------------------------------------

def spearman_rho(predicted_ranks, actual_ranks):
    """Spearman rank correlation."""
    n = len(predicted_ranks)
    if n < 2:
        return 0.0
    d2 = sum((p - a) ** 2 for p, a in zip(predicted_ranks, actual_ranks))
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


def assign_avg_ranks(scored):
    """Assign average ranks for tied scores.

    scored: list of (tmdb_id, score) sorted by score descending.
    Returns dict {tmdb_id: avg_rank}.
    """
    ranks = {}
    i = 0
    n = len(scored)
    while i < n:
        j = i
        # Find extent of tie group
        while j < n and scored[j][1] == scored[i][1]:
            j += 1
        # Average rank for positions i..j-1 (1-indexed)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[scored[k][0]] = avg_rank
        i = j
    return ranks


def _score_year(movie_components, movies, actual_ids, w_d, w_p, w_c, w_f, w_b=0.0):
    """Score a single year, returning (rho, t10) with proper tie handling."""
    scored = []
    for m in movies:
        tmdb_id = m["tmdb_id"]
        comps = movie_components.get((tmdb_id, m.get("_year", 0)))
        if comps is None:
            scored.append((tmdb_id, 0.0))
            continue
        d, p, c, f, b = comps
        scored.append((tmdb_id, w_d * d + w_p * p + w_c * c + w_f * f + w_b * b))

    # Sort by score descending, break ties randomly via tmdb_id hash
    scored.sort(key=lambda x: (x[1], hash(x[0])), reverse=True)
    pred_rank = assign_avg_ranks(scored)

    act_rank = {tid: rank + 1 for rank, tid in enumerate(actual_ids)}

    all_ids = [tid for tid, _ in scored]
    p_ranks = [pred_rank[tid] for tid in all_ids]
    a_ranks = [act_rank.get(tid, len(all_ids)) for tid in all_ids]
    rho = spearman_rho(p_ranks, a_ranks)

    # Top-10 overlap: for tied scores at the boundary, count conservatively
    # Use the sorted order (tie-broken by hash) for top-10 selection
    pred_top10 = set(tid for tid, _ in scored[:10])
    act_top10 = set(actual_ids[:10])
    t10 = len(pred_top10 & act_top10)

    return rho, t10


def evaluate(movie_components, backtest_movies_by_year, actual_ranks_by_year,
             w_d, w_p, w_c, w_f, w_b=0.0):
    """
    Compute objective = 0.6 * avg_rho + 0.4 * (avg_top10_overlap / 10).
    Returns (objective, avg_rho, avg_t10).
    """
    total_rho = 0.0
    total_t10 = 0
    n_years = 0

    for year in YEARS:
        movies = backtest_movies_by_year.get(year, [])
        if not movies:
            continue

        # Tag movies with year for component lookup
        tagged = [{**m, "_year": year} for m in movies]
        actual_ids = actual_ranks_by_year.get(year, [])
        rho, t10 = _score_year(movie_components, tagged, actual_ids,
                               w_d, w_p, w_c, w_f, w_b)

        total_rho += rho
        total_t10 += t10
        n_years += 1

    if n_years == 0:
        return 0.0, 0.0, 0.0

    avg_rho = total_rho / n_years
    avg_t10 = total_t10 / n_years
    objective = 0.6 * avg_rho + 0.4 * (avg_t10 / 10)
    return objective, avg_rho, avg_t10


def evaluate_per_year(movie_components, backtest_movies_by_year,
                      actual_ranks_by_year, w_d, w_p, w_c, w_f, w_b=0.0):
    """Return per-year (rho, t10) for reporting."""
    results = {}
    for year in YEARS:
        movies = backtest_movies_by_year.get(year, [])
        if not movies:
            continue
        tagged = [{**m, "_year": year} for m in movies]
        actual_ids = actual_ranks_by_year.get(year, [])
        rho, t10 = _score_year(movie_components, tagged, actual_ids,
                               w_d, w_p, w_c, w_f, w_b)
        results[year] = (rho, t10)
    return results


# ---------------------------------------------------------------------------
# Weight combinations that sum to 1.0
# ---------------------------------------------------------------------------

# Constraints: franchise weight (index 3) <= 0.50, talent sum (0+1+2) >= 0.40
MAX_FRANCHISE_WEIGHT = 0.50
MIN_TALENT_SUM = 0.40
# Recency: most recent movie must carry meaningful weight
MIN_RECENCY_FIRST = 0.5
# Consistency: spread between best (3 profitable) and worst (0) must be meaningful
MIN_CONSISTENCY_SPREAD = 0.3


# Director must retain meaningful weight (protects strong-director signals like Nolan)
MIN_DIRECTOR_WEIGHT = 0.15


def _valid_weights(combo):
    """Check weight constraints: franchise <= max, talent sum >= min, director >= min.

    Works for both 4-tuples (w_d, w_p, w_c, w_f) and 5-tuples (w_d, w_p, w_c, w_f, w_b).
    """
    return (combo[3] <= MAX_FRANCHISE_WEIGHT + 1e-9
            and sum(combo[:3]) >= MIN_TALENT_SUM - 1e-9
            and combo[0] >= MIN_DIRECTOR_WEIGHT - 1e-9)


def weight_combos(step, n=5):
    """Generate all n-tuples of non-negative multiples of step that sum to 1.0."""
    steps = int(round(1.0 / step))
    combos = []
    for combo in product(range(steps + 1), repeat=n):
        if sum(combo) == steps:
            w = tuple(c * step for c in combo)
            if _valid_weights(w):
                combos.append(w)
    return combos


def weight_combos_around(center, step, radius, n=5):
    """Generate weight combos near center, within radius, summing to 1.0."""
    steps_per_unit = round(1.0 / step)
    combos = []
    lo = [max(0, int(round((c - radius) / step))) for c in center]
    hi = [min(steps_per_unit, int(round((c + radius) / step))) for c in center]
    for combo in product(*(range(lo[i], hi[i] + 1) for i in range(n))):
        if sum(combo) == steps_per_unit:
            w = tuple(c * step for c in combo)
            if _valid_weights(w):
                combos.append(w)
    return combos


def descending_combos(n, lo, hi, step):
    """Generate n-tuples where values are descending, each in [lo, hi]."""
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 4))
        v += step
    combos = []
    _desc_helper(vals, n, [], combos)
    return combos


def _desc_helper(vals, n, current, results):
    if len(current) == n:
        results.append(tuple(current))
        return
    for v in vals:
        if not current or v <= current[-1]:
            current.append(v)
            _desc_helper(vals, n, current, results)
            current.pop()


def descending_combos_around(center, step, radius, lo, hi):
    """Generate descending combos near center within radius."""
    ranges = []
    for c in center:
        r_lo = max(lo, round(c - radius, 4))
        r_hi = min(hi, round(c + radius, 4))
        vals = []
        v = r_lo
        while v <= r_hi + 1e-9:
            vals.append(round(v, 4))
            v += step
        ranges.append(vals)

    combos = []
    _desc_around_helper(ranges, 0, [], combos)
    return combos


def _desc_around_helper(ranges, idx, current, results):
    if idx == len(ranges):
        results.append(tuple(current))
        return
    for v in ranges[idx]:
        if not current or v <= current[-1]:
            current.append(v)
            _desc_around_helper(ranges, idx + 1, current, results)
            current.pop()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize(person_history, movie_credits_map, backtest_movies_by_year,
             actual_ranks_by_year, buzz_lookup=None):
    """Run coordinate descent optimization. Returns best params dict."""

    # 5 weights: director, producer, cast, franchise, buzz
    best_weights = (0.20, 0.15, 0.15, 0.40, 0.10)
    best_franchise = 50.0
    best_recency = [1.0, 0.7, 0.4]
    best_consistency = {3: 1.2, 2: 1.0, 1: 0.8, 0: 0.5}

    # Compute baseline
    ps = compute_person_scores(person_history, best_recency, best_consistency)
    mc = compute_movie_components(ps, movie_credits_map, backtest_movies_by_year,
                                  best_franchise)
    baseline_obj, baseline_rho, baseline_t10 = evaluate(
        mc, backtest_movies_by_year, actual_ranks_by_year, *best_weights
    )

    print(f"  Baseline: obj={baseline_obj:.4f} (rho={baseline_rho:.3f}, t10={baseline_t10:.1f})")

    # -----------------------------------------------------------------------
    # Phase 1: Coarse coordinate descent
    # -----------------------------------------------------------------------
    print("\n  Phase 1: Coarse search...")
    best_obj = baseline_obj
    prev_obj = -1.0

    for iteration in range(1, 16):
        if abs(best_obj - prev_obj) < 0.001 and iteration > 1:
            print(f"    Converged after {iteration - 1} iterations.")
            break
        prev_obj = best_obj

        # Step 1: Component weights (Layer 4 only)
        combos = weight_combos(0.10)
        for w in combos:
            obj, rho, t10 = evaluate(mc, backtest_movies_by_year,
                                     actual_ranks_by_year, *w)
            if obj > best_obj:
                best_obj = obj
                best_weights = w

        # Step 2: Franchise bonus (Layer 3+4)
        for fb in range(0, 201, 10):
            fb_f = float(fb)
            mc_test = compute_movie_components(
                ps, movie_credits_map, backtest_movies_by_year, fb_f, buzz_lookup
            )
            obj, rho, t10 = evaluate(mc_test, backtest_movies_by_year,
                                     actual_ranks_by_year, *best_weights)
            if obj > best_obj:
                best_obj = obj
                best_franchise = fb_f
                mc = mc_test

        # Recompute mc with best franchise
        mc = compute_movie_components(ps, movie_credits_map,
                                      backtest_movies_by_year, best_franchise, buzz_lookup)

        # Step 3: Recency weights (Layer 2+3+4)
        rec_combos = [rw for rw in descending_combos(3, 0.0, 1.0, 0.1)
                      if rw[0] >= MIN_RECENCY_FIRST]
        for rw in rec_combos:
            ps_test = compute_person_scores(person_history, list(rw),
                                            best_consistency)
            mc_test = compute_movie_components(
                ps_test, movie_credits_map, backtest_movies_by_year,
                best_franchise
            )
            obj, rho, t10 = evaluate(mc_test, backtest_movies_by_year,
                                     actual_ranks_by_year, *best_weights)
            if obj > best_obj:
                best_obj = obj
                best_recency = list(rw)
                ps = ps_test
                mc = mc_test

        # Recompute ps/mc with best recency
        ps = compute_person_scores(person_history, best_recency, best_consistency)
        mc = compute_movie_components(ps, movie_credits_map,
                                      backtest_movies_by_year, best_franchise, buzz_lookup)

        # Step 4: Consistency map (Layer 2+3+4)
        cons_combos = [cc for cc in descending_combos(4, 0.2, 2.0, 0.2)
                       if cc[0] - cc[3] >= MIN_CONSISTENCY_SPREAD]
        for cc in cons_combos:
            cmap = {3: cc[0], 2: cc[1], 1: cc[2], 0: cc[3]}
            ps_test = compute_person_scores(person_history, best_recency, cmap)
            mc_test = compute_movie_components(
                ps_test, movie_credits_map, backtest_movies_by_year,
                best_franchise
            )
            obj, rho, t10 = evaluate(mc_test, backtest_movies_by_year,
                                     actual_ranks_by_year, *best_weights)
            if obj > best_obj:
                best_obj = obj
                best_consistency = cmap
                ps = ps_test
                mc = mc_test

        # Recompute ps/mc with best consistency
        ps = compute_person_scores(person_history, best_recency, best_consistency)
        mc = compute_movie_components(ps, movie_credits_map,
                                      backtest_movies_by_year, best_franchise, buzz_lookup)

        obj, rho, t10 = evaluate(mc, backtest_movies_by_year,
                                 actual_ranks_by_year, *best_weights)
        print(f"    Iteration {iteration}: obj={obj:.4f} (rho={rho:.3f}, t10={t10:.1f})")

    # -----------------------------------------------------------------------
    # Phase 2: Fine refinement
    # -----------------------------------------------------------------------
    print("\n  Phase 2: Fine refinement...")

    # Fine weights
    fine_w = weight_combos_around(best_weights, 0.05, 0.15)
    for w in fine_w:
        obj, rho, t10 = evaluate(mc, backtest_movies_by_year,
                                 actual_ranks_by_year, *w)
        if obj > best_obj:
            best_obj = obj
            best_weights = w

    # Fine franchise bonus
    fb_lo = max(0, best_franchise - 20)
    fb_hi = best_franchise + 20
    fb = fb_lo
    while fb <= fb_hi + 0.01:
        mc_test = compute_movie_components(
            ps, movie_credits_map, backtest_movies_by_year, fb, buzz_lookup
        )
        obj, rho, t10 = evaluate(mc_test, backtest_movies_by_year,
                                 actual_ranks_by_year, *best_weights)
        if obj > best_obj:
            best_obj = obj
            best_franchise = fb
            mc = mc_test
        fb += 2.0

    mc = compute_movie_components(ps, movie_credits_map,
                                  backtest_movies_by_year, best_franchise, buzz_lookup)

    # Fine recency weights
    fine_rec = [rw for rw in descending_combos_around(best_recency, 0.05, 0.15, 0.0, 1.0)
                 if rw[0] >= MIN_RECENCY_FIRST]
    for rw in fine_rec:
        ps_test = compute_person_scores(person_history, list(rw), best_consistency)
        mc_test = compute_movie_components(
            ps_test, movie_credits_map, backtest_movies_by_year, best_franchise
        )
        obj, rho, t10 = evaluate(mc_test, backtest_movies_by_year,
                                 actual_ranks_by_year, *best_weights)
        if obj > best_obj:
            best_obj = obj
            best_recency = list(rw)
            ps = ps_test
            mc = mc_test

    ps = compute_person_scores(person_history, best_recency, best_consistency)
    mc = compute_movie_components(ps, movie_credits_map,
                                  backtest_movies_by_year, best_franchise, buzz_lookup)

    # Fine consistency map
    center_cons = [best_consistency[3], best_consistency[2],
                   best_consistency[1], best_consistency[0]]
    fine_cons = [cc for cc in descending_combos_around(center_cons, 0.05, 0.2, 0.2, 2.0)
                  if cc[0] - cc[3] >= MIN_CONSISTENCY_SPREAD]
    for cc in fine_cons:
        cmap = {3: cc[0], 2: cc[1], 1: cc[2], 0: cc[3]}
        ps_test = compute_person_scores(person_history, best_recency, cmap)
        mc_test = compute_movie_components(
            ps_test, movie_credits_map, backtest_movies_by_year, best_franchise
        )
        obj, rho, t10 = evaluate(mc_test, backtest_movies_by_year,
                                 actual_ranks_by_year, *best_weights)
        if obj > best_obj:
            best_obj = obj
            best_consistency = cmap
            ps = ps_test
            mc = mc_test

    ps = compute_person_scores(person_history, best_recency, best_consistency)
    mc = compute_movie_components(ps, movie_credits_map,
                                  backtest_movies_by_year, best_franchise, buzz_lookup)

    final_obj, final_rho, final_t10 = evaluate(
        mc, backtest_movies_by_year, actual_ranks_by_year, *best_weights
    )
    print(f"    Final: obj={final_obj:.4f} (rho={final_rho:.3f}, t10={final_t10:.1f})")

    return {
        "weights": best_weights,
        "franchise_bonus": best_franchise,
        "recency": best_recency,
        "consistency": best_consistency,
        "objective": final_obj,
        "rho": final_rho,
        "t10": final_t10,
        "movie_components": mc,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(baseline, optimized, backtest_movies_by_year,
                 actual_ranks_by_year, buzz_lookup=None):
    """Print comparison report."""
    b = baseline
    o = optimized

    print(f"\n  {'Parameter':<20} {'Baseline':>12} {'Optimized':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    labels = ["W_DIRECTOR", "W_PRODUCER", "W_CAST", "W_FRANCHISE", "W_BUZZ"]
    for i, label in enumerate(labels):
        print(f"  {label:<20} {b['weights'][i]:>12.3f} {o['weights'][i]:>12.3f}")
    print(f"  {'FRANCHISE_BONUS':<20} {b['franchise_bonus']:>12.1f} {o['franchise_bonus']:>12.1f}")
    for i, label in enumerate(["RECENCY_W1", "RECENCY_W2", "RECENCY_W3"]):
        print(f"  {label:<20} {b['recency'][i]:>12.2f} {o['recency'][i]:>12.2f}")
    for k in [3, 2, 1, 0]:
        label = f"CONSISTENCY_{k}"
        print(f"  {label:<20} {b['consistency'][k]:>12.2f} {o['consistency'][k]:>12.2f}")

    # Per-year comparison
    # Recompute baseline per-year
    b_ps = compute_person_scores(
        baseline["person_history"], b["recency"], b["consistency"]
    )
    b_mc = compute_movie_components(
        b_ps, baseline["movie_credits_map"], backtest_movies_by_year,
        b["franchise_bonus"], buzz_lookup
    )
    b_per_year = evaluate_per_year(
        b_mc, backtest_movies_by_year, actual_ranks_by_year, *b["weights"]
    )
    o_per_year = evaluate_per_year(
        o["movie_components"], backtest_movies_by_year, actual_ranks_by_year,
        *o["weights"]
    )

    print(f"\n  {'Year':>4}  {'Rho(old)':>8}  {'Rho(new)':>8}  {'T10(old)':>8}  {'T10(new)':>8}")
    print(f"  {'----':>4}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
    sum_b_rho = sum_o_rho = 0.0
    sum_b_t10 = sum_o_t10 = 0
    count = 0
    for year in YEARS:
        if year in b_per_year and year in o_per_year:
            br, bt = b_per_year[year]
            opr, ot = o_per_year[year]
            print(f"  {year:>4}  {br:>8.3f}  {opr:>8.3f}  {bt:>8}  {ot:>8}")
            sum_b_rho += br
            sum_o_rho += opr
            sum_b_t10 += bt
            sum_o_t10 += ot
            count += 1

    if count:
        print(f"  {'----':>4}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
        print(f"  {'AVG':>4}  {sum_b_rho/count:>8.3f}  {sum_o_rho/count:>8.3f}"
              f"  {sum_b_t10/count:>8.1f}  {sum_o_t10/count:>8.1f}")

    b_obj = b["objective"]
    o_obj = o["objective"]
    pct = ((o_obj - b_obj) / b_obj * 100) if b_obj > 0 else 0.0
    print(f"\n  Objective: {b_obj:.4f} -> {o_obj:.4f} ({pct:+.1f}%)")


# ---------------------------------------------------------------------------
# Update scoring.py in-place
# ---------------------------------------------------------------------------

def update_scoring_py(optimized):
    """Update constants in scoring.py using regex substitution."""
    with open(SCORING_PATH, "r") as f:
        content = f.read()

    w_d, w_p, w_c, w_f, w_b = [round(x, 4) for x in optimized["weights"]]
    rec = [round(x, 4) for x in optimized["recency"]]
    cons = {k: round(v, 4) for k, v in optimized["consistency"].items()}
    fb = round(optimized["franchise_bonus"], 1)

    # Update RECENCY_WEIGHTS
    content = re.sub(
        r"RECENCY_WEIGHTS\s*=\s*\[.*?\]",
        f"RECENCY_WEIGHTS = [{rec[0]}, {rec[1]}, {rec[2]}]",
        content,
    )

    # Update CONSISTENCY_MAP
    content = re.sub(
        r"CONSISTENCY_MAP\s*=\s*\{[^}]*\}",
        f"CONSISTENCY_MAP = {{\n"
        f"    3: {cons[3]},\n"
        f"    2: {cons[2]},\n"
        f"    1: {cons[1]},\n"
        f"    0: {cons[0]},\n"
        f"}}",
        content,
        flags=re.DOTALL,
    )

    # Update component weights
    content = re.sub(
        r"WEIGHT_DIRECTOR\s*=\s*[\d.]+",
        f"WEIGHT_DIRECTOR = {w_d}",
        content,
    )
    content = re.sub(
        r"WEIGHT_PRODUCER\s*=\s*[\d.]+",
        f"WEIGHT_PRODUCER = {w_p}",
        content,
    )
    content = re.sub(
        r"WEIGHT_CAST\s*=\s*[\d.]+",
        f"WEIGHT_CAST = {w_c}",
        content,
    )
    content = re.sub(
        r"WEIGHT_FRANCHISE\s*=\s*[\d.]+",
        f"WEIGHT_FRANCHISE = {w_f}",
        content,
    )
    content = re.sub(
        r"WEIGHT_BUZZ\s*=\s*[\d.]+",
        f"WEIGHT_BUZZ = {w_b}",
        content,
    )

    # Update franchise bonus (the 50.0 in compute_franchise_bonus)
    content = re.sub(
        r"(# A known franchise gets a base bonus of )\d+(\s*\(in score units\)\.)\n(\s*bonus = )[\d.]+",
        rf"\g<1>{int(fb)}\g<2>\n\g<3>{fb}",
        content,
    )

    with open(SCORING_PATH, "w") as f:
        f.write(content)

    print(f"\n  Updated {SCORING_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  WEIGHT OPTIMIZER -- BOX OFFICE SCORING ALGORITHM")
    print("=" * 64)

    print("\n  Loading cached data...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    movie_lookup, filmography_lookup, credits_lookup, backtest_movies_by_year, buzz_lookup = \
        load_all_data(conn)
    conn.close()

    print(f"    Movies: {len(movie_lookup)}, People: {len(filmography_lookup)}, "
          f"Credits: {len(credits_lookup)}, Years: {len(backtest_movies_by_year)}, "
          f"Buzz entries: {len(buzz_lookup)}")

    print("\n  Pre-computing Layer 1...")
    person_history, movie_credits_map, actual_ranks_by_year = precompute_layer1(
        movie_lookup, filmography_lookup, credits_lookup, backtest_movies_by_year,
        buzz_lookup
    )
    print(f"    Person-year pairs: {len(person_history)}, "
          f"Movie-year pairs: {len(movie_credits_map)}")

    # Baseline params (5 weights: dir, prod, cast, franchise, buzz)
    baseline = {
        "weights": (0.20, 0.15, 0.15, 0.40, 0.10),
        "franchise_bonus": 50.0,
        "recency": [1.0, 0.7, 0.4],
        "consistency": {3: 1.2, 2: 1.0, 1: 0.8, 0: 0.5},
        "person_history": person_history,
        "movie_credits_map": movie_credits_map,
    }

    # Compute baseline objective
    b_ps = compute_person_scores(person_history, baseline["recency"],
                                 baseline["consistency"])
    b_mc = compute_movie_components(b_ps, movie_credits_map,
                                    backtest_movies_by_year,
                                    baseline["franchise_bonus"], buzz_lookup)
    b_obj, b_rho, b_t10 = evaluate(b_mc, backtest_movies_by_year,
                                   actual_ranks_by_year, *baseline["weights"])
    baseline["objective"] = b_obj
    baseline["rho"] = b_rho
    baseline["t10"] = b_t10

    # Run optimizer
    optimized = optimize(person_history, movie_credits_map,
                         backtest_movies_by_year, actual_ranks_by_year,
                         buzz_lookup)

    # Report
    print_report(baseline, optimized, backtest_movies_by_year,
                 actual_ranks_by_year, buzz_lookup)

    # Update scoring.py
    update_scoring_py(optimized)

    print("\n  Done. Run 'python backtest.py' to verify improved metrics.")
    print("=" * 64)


if __name__ == "__main__":
    main()
