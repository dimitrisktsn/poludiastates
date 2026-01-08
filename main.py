from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Set, List, Tuple, Dict

import pandas as pd


import numpy as np
from kd_tree import KDTree
from quad_tree import QuadTree
from range_tree import RangeTree
from r_tree import RTree
from lsh_text import MinHashLSH


#query
@dataclass
class AssignmentQueryParams:

    "Παράμετροι ερωτήματος (αριθμητικά και κατηγορικά φίλτρα)."
    year_min: int = 2000
    year_max: int = 2020
    pop_min: float = 8.0
    pop_max: float = 12.0
    vote_min: float = 3.0
    vote_max: float = 8.0
    runtime_min: float = 30.0
    runtime_max: float = 180.0
    allowed_countries: tuple[str, ...] = ("US", "GB")
    language: str = "en"


#parsing ta dedomena
def parse_origin_country(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        return [str(val)]


def parse_list_field(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        return [str(val)]


def load_dataset() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "data_movies_clean.xlsx"
    df = pd.read_excel(file_path)
    print(f"[DATA] Raw rows: {len(df)}")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    numeric_cols = ["release_year", "popularity", "vote_average", "runtime", "vote_count"]
    df = df.dropna(subset=numeric_cols)

    df["origin_country_parsed"] = df["origin_country"].apply(parse_origin_country)
    df["genre_list"] = df["genre_names"].apply(parse_list_field)
    df["production_company_list"] = df["production_company_names"].apply(parse_list_field)
    df["original_language_lower"] = df["original_language"].astype(str).str.lower()

    print(f"[DATA] Rows after preprocessing: {len(df)}")
    return df


def build_base_pool(df: pd.DataFrame) -> pd.DataFrame:

    "Δημιουργεί το βασικό σύνολο δεδομένων με τα γενικά φίλτρα."
    mask_lang = df["original_language_lower"] == "en"
    mask_runtime = df["runtime"].between(30, 240)
    mask_votes = df["vote_count"] >= 10

    base_df = df[mask_lang & mask_runtime & mask_votes].copy()
    print(f"[DATA] Base pool size: {len(base_df)}")
    return base_df


def apply_full_query_filters(df: pd.DataFrame, params: AssignmentQueryParams) -> pd.DataFrame:
    
    "Εφαρμόζει όλα τα φίλτρα του ερωτήματος με χρήση pandas."
    if df.empty:
        return df.copy()

    mask_year = df["release_year"].between(params.year_min, params.year_max)
    mask_pop = df["popularity"].between(params.pop_min, params.pop_max)
    mask_vote = df["vote_average"].between(params.vote_min, params.vote_max)
    mask_runtime = df["runtime"].between(params.runtime_min, params.runtime_max)

    if params.allowed_countries:
        mask_country = df["origin_country_parsed"].apply(
            lambda lst: any(c in params.allowed_countries for c in lst)
        )
    else:
        mask_country = pd.Series([True] * len(df), index=df.index)

    mask_lang = df["original_language_lower"] == params.language.lower()

    full_mask = mask_year & mask_pop & mask_vote & mask_runtime & mask_country & mask_lang
    return df[full_mask].copy()


#measure to build time + index
def build_kd_index(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[KDTree, float]:
    feature_matrix = df[feature_cols].to_numpy()
    t0 = time.perf_counter()
    kd_tree = KDTree(feature_matrix)
    t1 = time.perf_counter()
    build_time = t1 - t0
    return kd_tree, build_time


def build_quad_index(df: pd.DataFrame) -> Tuple[QuadTree, float]:
    quad_points = df[["popularity", "vote_average"]].to_numpy()
    t0 = time.perf_counter()
    qtree = QuadTree(quad_points, capacity=4)
    t1 = time.perf_counter()
    build_time = t1 - t0
    return qtree, build_time


def build_range_index(df: pd.DataFrame) -> Tuple[RangeTree, float]:
    years = df["release_year"].to_numpy()
    t0 = time.perf_counter()
    year_tree = RangeTree(years)
    t1 = time.perf_counter()
    build_time = t1 - t0
    return year_tree, build_time


def build_rtree_index(df: pd.DataFrame) -> Tuple[RTree, float]:
    rtree_points = df[["popularity", "vote_average"]].to_numpy()
    t0 = time.perf_counter()
    rtree = RTree(max_entries=16)
    for idx, (pop, vote) in enumerate(rtree_points):
        bbox = (float(pop), float(vote), float(pop), float(vote))
        rtree.insert(idx, bbox)
    t1 = time.perf_counter()
    build_time = t1 - t0
    return rtree, build_time


#numeric filtering
def numeric_candidates_kdtree(
    kd_tree: KDTree,
    df: pd.DataFrame,
    params: AssignmentQueryParams,
    feature_cols: list[str],
) -> Tuple[pd.DataFrame, float]:
    vc_min = float(df["vote_count"].min())
    vc_max = float(df["vote_count"].max())

    lower_bounds = [
        params.year_min,
        params.pop_min,
        params.vote_min,
        params.runtime_min,
        vc_min,
    ]
    upper_bounds = [
        params.year_max,
        params.pop_max,
        params.vote_max,
        params.runtime_max,
        vc_max,
    ]

    t0 = time.perf_counter()
    indices = kd_tree.range_query(lower_bounds, upper_bounds)
    t1 = time.perf_counter()
    query_time = t1 - t0

    candidates = df.iloc[indices].copy()
    return candidates, query_time


def numeric_candidates_quadtree(
    quad_tree: QuadTree,
    df: pd.DataFrame,
    params: AssignmentQueryParams,
) -> Tuple[pd.DataFrame, float]:
    rect = (params.pop_min, params.vote_min, params.pop_max, params.vote_max)
    t0 = time.perf_counter()
    quad_indices = quad_tree.range_query(rect)
    t1 = time.perf_counter()
    query_time = t1 - t0

    quad_indices = sorted(set(quad_indices))
    candidates = df.iloc[quad_indices].copy()
    return candidates, query_time


def numeric_candidates_rangetree(
    year_tree: RangeTree,
    df: pd.DataFrame,
    params: AssignmentQueryParams,
) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    year_indices = year_tree.range_query(params.year_min, params.year_max)
    t1 = time.perf_counter()
    query_time = t1 - t0
    candidates = df.iloc[year_indices].copy()
    return candidates, query_time


def numeric_candidates_rtree(
    rtree: RTree,
    df: pd.DataFrame,
    params: AssignmentQueryParams,
) -> Tuple[pd.DataFrame, float]:
    query_rect = (params.pop_min, params.vote_min, params.pop_max, params.vote_max)
    t0 = time.perf_counter()
    rtree_indices = rtree.range_query(query_rect)
    t1 = time.perf_counter()
    query_time = t1 - t0
    candidates = df.iloc[rtree_indices].copy()
    return candidates, query_time


def brute_force_knn(
    df: pd.DataFrame,
    feature_cols: list[str],
    query_point: list[float],
    k: int = 10,
) -> List[int]:
    
    "Απλή brute-force υλοποίηση kNN για σύγκριση με το KD-Tree."
    if df.empty:
        return []

    X = df[feature_cols].to_numpy(dtype=float)
    q = np.array(query_point, dtype=float)
    d2 = np.sum((X - q) ** 2, axis=1)

    k = min(k, len(df))
    idx = np.argpartition(d2, k - 1)[:k]
    idx = idx[np.argsort(d2[idx])]
    return idx.tolist()


#LSH
def genres_to_token_set(genres: List[str]) -> Set[str]:
    return {str(g).strip().lower() for g in genres if str(g).strip()}


@dataclass
class GenreLSHIndex:
    
    "Δομή που περιέχει το LSH και τα δεδομένα genres."
    lsh: MinHashLSH
    docs: List[Set[str]]
    df: pd.DataFrame


def build_genre_lsh_index(df: pd.DataFrame) -> GenreLSHIndex:
    
    "Κατασκευάζει LSH index για τα genres του dataset."
    df_local = df.reset_index(drop=True)
    docs: List[Set[str]] = []

    for _, row in df_local.iterrows():
        genres = row.get("genre_list", [])
        if not isinstance(genres, list):
            genres = []
        docs.append(genres_to_token_set(genres))

    lsh = MinHashLSH(num_perm=128, num_bands=16, seed=42, fallback_all=False)
    lsh.fit(docs)

    return GenreLSHIndex(lsh=lsh, docs=docs, df=df_local)


def extract_all_genres(df: pd.DataFrame) -> List[str]:
    
    "Επιστρέφει όλα τα μοναδικά genres του dataset."
    all_genres: Set[str] = set()

    if "genre_list" not in df.columns:
        return []

    for lst in df["genre_list"]:
        if not isinstance(lst, list):
            continue
        for g in lst:
            g_clean = str(g).strip()
            if g_clean:
                all_genres.add(g_clean)

    return sorted(all_genres)


def build_lsh_on_genres(df: pd.DataFrame) -> Tuple[MinHashLSH, List[Set[str]], float]:
   
    "Κατασκευάζει LSH πάνω στα genres και μετρά τον χρόνο."
    df_local = df.reset_index(drop=True)
    docs: List[Set[str]] = []
    for _, row in df_local.iterrows():
        genres = row.get("genre_list", [])
        if not isinstance(genres, list):
            genres = []
        docs.append(genres_to_token_set(genres))

    t0 = time.perf_counter()
    lsh = MinHashLSH(num_perm=64, num_bands=8, seed=42)
    lsh.fit(docs)
    t1 = time.perf_counter()
    build_time = t1 - t0

    return lsh, docs, build_time


def run_lsh_on_df(
    df: pd.DataFrame,
    label: str,
    N: int = 3,
) -> Tuple[float, float]:
    
    "Εκτελεί LSH σε dataframe και επιστρέφει χρόνους εκτέλεσης."
    if df.empty:
        return 0.0, 0.0

    df_local = df.reset_index(drop=True)
    lsh, docs, build_time = build_lsh_on_genres(df_local)

    #vres mia tainia me mi kina g enres gia na kanoume 1 query
    query_idx = None
    for i, s in enumerate(docs):
        if len(s) > 0:
            query_idx = i
            break

    if query_idx is None:
        #oles oi tainies xoris genres -> LSH axristo
        return build_time, 0.0

    query_set = docs[query_idx]

    #metrame mono ton xrono edo
    t0 = time.perf_counter()
    _ = lsh.query(query_set)
    t1 = time.perf_counter()
    query_time = t1 - t0

    return build_time, query_time


def run_lsh_on_ground_truth(df: pd.DataFrame, N: int = 3) -> Tuple[float, float]:
    """
    Wrapper για συμβατότητα με GUI κ.λπ.
    Τρέχει LSH πάνω στο df (ground truth ή όποιο subset).
    """
    return run_lsh_on_df(df, "GROUND_TRUTH", N=N)


def query_by_genre_name(
    genre_index: GenreLSHIndex,
    genre_name: str,
    top_n: int = 5,
) -> pd.DataFrame:
    
    "Επιστρέφει τις πιο σχετικές ταινίες για συγκεκριμένο genre με LSH."
    genre_name = str(genre_name).strip()
    if not genre_name:
        return genre_index.df.iloc[0:0].copy()

    query_set = genres_to_token_set([genre_name])

    raw_results = genre_index.lsh.query(query_set)
    if not raw_results:
        return genre_index.df.iloc[0:0].copy()

    first = raw_results[0]
    if isinstance(first, tuple):
        idxs = [r[0] for r in raw_results]
    else:
        idxs = list(raw_results)

    idxs = [i for i in idxs if 0 <= i < len(genre_index.df)]
    idxs = idxs[:top_n]

    return genre_index.df.iloc[idxs].copy()


#gia na fainontai ta dedomena
def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    AssignmentQueryParams,
    List[str],
    GenreLSHIndex,
]:
    "Προετοιμάζει τα δεδομένα και τα indexes για την εκτέλεση των queries."
    "Επιστρέφει τα βασικά dataframes, παραμέτρους και LSH index."

    df_raw = load_dataset()
    df_processed = preprocess_dataset(df_raw)
    base_df = build_base_pool(df_processed)

    params = AssignmentQueryParams()
    ground_truth_df = apply_full_query_filters(base_df, params)
    print(f"[GT] Ground-truth result size: {len(ground_truth_df)}")

    all_genres = extract_all_genres(ground_truth_df)
    print(f"[INFO] Unique genres: {len(all_genres)}")

    genre_index = build_genre_lsh_index(ground_truth_df)

    return df_processed, base_df, ground_truth_df, params, all_genres, genre_index


def evaluate_indexes(
    base_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    params: AssignmentQueryParams,
):
    summary_numeric: Dict[str, Dict[str, float | int]] = {}
    summary_schemes: Dict[str, Dict[str, float | int]] = {}

    #KD TREE
    kd_feature_cols = ["release_year", "popularity", "vote_average", "runtime", "vote_count"]
    kd_tree, kd_build_time = build_kd_index(base_df, kd_feature_cols)
    kd_numeric_df, kd_numeric_time = numeric_candidates_kdtree(kd_tree, base_df, params, kd_feature_cols)

    #kNN (μόνο για KD-Tree)
    #query point: κέντρο των numeric ranges + median(vote_count)
    vc_med = float(base_df["vote_count"].median())
    knn_query_point = [
        (params.year_min + params.year_max) / 2,
        (params.pop_min + params.pop_max) / 2,
        (params.vote_min + params.vote_max) / 2,
        (params.runtime_min + params.runtime_max) / 2,
        vc_med,
    ]

    t0 = time.perf_counter()
    _ = kd_tree.knn_query(knn_query_point, k=10)
    t1 = time.perf_counter()
    kd_knn_time = t1 - t0

    #Brute-force kNN baseline (ίδιο query)
    t0 = time.perf_counter()
    _ = brute_force_knn(base_df, kd_feature_cols, knn_query_point, k=10)
    t1 = time.perf_counter()
    brute_knn_time = t1 - t0

    summary_numeric["KD-Tree"] = {
        "build": kd_build_time,
        "numeric": kd_numeric_time,
        "numeric_candidates": len(kd_numeric_df),
        "knn": kd_knn_time,
        "knn_bruteforce": brute_knn_time,
    }

    #QUAD TREE
    quad_tree, quad_build_time = build_quad_index(base_df)
    quad_numeric_df, quad_numeric_time = numeric_candidates_quadtree(quad_tree, base_df, params)
    summary_numeric["Quad-Tree"] = {
        "build": quad_build_time,
        "numeric": quad_numeric_time,
        "numeric_candidates": len(quad_numeric_df),
    }

    #RANGE TREE
    year_tree, range_build_time = build_range_index(base_df)
    range_numeric_df, range_numeric_time = numeric_candidates_rangetree(year_tree, base_df, params)
    summary_numeric["Range-Tree"] = {
        "build": range_build_time,
        "numeric": range_numeric_time,
        "numeric_candidates": len(range_numeric_df),
    }

    #R TREE
    rtree, rtree_build_time = build_rtree_index(base_df)
    rtree_numeric_df, rtree_numeric_time = numeric_candidates_rtree(rtree, base_df, params)
    summary_numeric["R-Tree"] = {
        "build": rtree_build_time,
        "numeric": rtree_numeric_time,
        "numeric_candidates": len(rtree_numeric_df),
    }

    #lsh pano sto ground truth gia koino metro
    lsh_common_build, lsh_common_query = run_lsh_on_df(ground_truth_df, "COMMON", N=3)

    #KD TREE + LSH
    kd_scheme_df = apply_full_query_filters(kd_numeric_df, params)
    kd_lsh_build, kd_lsh_query = run_lsh_on_df(kd_scheme_df, "KD-Tree + LSH", N=3)
    summary_schemes["KD-Tree + LSH"] = {
        "build_index": kd_build_time,
        "numeric_time": kd_numeric_time,
        "numeric_candidates": len(kd_numeric_df),
        "result_size": len(kd_scheme_df),
        "lsh_build": kd_lsh_build,
        "lsh_query": kd_lsh_query,
    }

    #QUAD TREE + LSH
    quad_scheme_df = apply_full_query_filters(quad_numeric_df, params)
    quad_lsh_build, quad_lsh_query = run_lsh_on_df(quad_scheme_df, "Quad-Tree + LSH", N=3)
    summary_schemes["Quad-Tree + LSH"] = {
        "build_index": quad_build_time,
        "numeric_time": quad_numeric_time,
        "numeric_candidates": len(quad_numeric_df),
        "result_size": len(quad_scheme_df),
        "lsh_build": quad_lsh_build,
        "lsh_query": quad_lsh_query,
    }

    #RANGE TREE + LSH
    range_scheme_df = apply_full_query_filters(range_numeric_df, params)
    range_lsh_build, range_lsh_query = run_lsh_on_df(range_scheme_df, "Range-Tree + LSH", N=3)
    summary_schemes["Range-Tree + LSH"] = {
        "build_index": range_build_time,
        "numeric_time": range_numeric_time,
        "numeric_candidates": len(range_numeric_df),
        "result_size": len(range_scheme_df),
        "lsh_build": range_lsh_build,
        "lsh_query": range_lsh_query,
    }

    #R TREE + LSH
    rtree_scheme_df = apply_full_query_filters(rtree_numeric_df, params)
    rtree_lsh_build, rtree_lsh_query = run_lsh_on_df(rtree_scheme_df, "R-Tree + LSH", N=3)
    summary_schemes["R-Tree + LSH"] = {
        "build_index": rtree_build_time,
        "numeric_time": rtree_numeric_time,
        "numeric_candidates": len(rtree_numeric_df),
        "result_size": len(rtree_scheme_df),
        "lsh_build": rtree_lsh_build,
        "lsh_query": rtree_lsh_query,
    }

    return summary_numeric, summary_schemes, lsh_common_build, lsh_common_query


def print_summaries(
    summary_numeric: Dict[str, Dict[str, float | int]],
    summary_schemes: Dict[str, Dict[str, float | int]],
    lsh_common_build: float,
    lsh_common_query: float,
    gt_size: int,
):

    print("\n")
    print("NUMERIC INDEX PERFORMANCE (seconds)")
    print(f"{'Index':<12} {'Build':>8} {'NumericQ':>10} {'kNN':>8} {'kNN-BF':>8} {'Cand':>8}")
    for name, stats in summary_numeric.items():
        print(
            f"{name:<12} "
            f"{stats['build']:8.4f} "
            f"{stats['numeric']:10.4f} "
            f"{stats.get('knn', float('nan')):8.4f} "
            f"{stats.get('knn_bruteforce', float('nan')):8.4f} "
            f"{stats['numeric_candidates']:8d}"
        )

    print("\n")
    print("SCHEMES: (Index + LSH) PERFORMANCE")
    print(f"{'Scheme':<16} {'IdxBuild':>8} {'NumQ':>8} {'Cand':>8} {'Res':>8} {'LSHbuild':>9} {'LSHq':>8}")
    for name, stats in summary_schemes.items():
        print(
            f"{name:<16} "
            f"{stats['build_index']:8.4f} "
            f"{stats['numeric_time']:8.4f} "
            f"{stats['numeric_candidates']:8d} "
            f"{stats['result_size']:8d} "
            f"{stats['lsh_build']:9.4f} "
            f"{stats['lsh_query']:8.6f}"
        )

    print("\n[COMMON LSH on global ground truth]")
    print(f"  Build time : {lsh_common_build:.4f} s  (|GT|={gt_size})")
    print(f"  Query time : {lsh_common_query:.6f} s")


#i main
def main():
    #proetoimasia data + genres + global genre LSH
    df_processed, base_df, ground_truth_df, params, all_genres, genre_index = prepare_data()

    #axiologisi domon
    summary_numeric, summary_schemes, lsh_common_build, lsh_common_query = evaluate_indexes(
        base_df,
        ground_truth_df,
        params,
    )

    #print perilipseon
    print_summaries(summary_numeric, summary_schemes, lsh_common_build, lsh_common_query, len(ground_truth_df))

    #search me ena genre (proairetiko)
    if all_genres:
        demo_genre = all_genres[0]
        print(f"\n[DEMO] Top-5 movies for genre: {demo_genre}")
        demo_results = query_by_genre_name(genre_index, demo_genre, top_n=5)
        cols = [c for c in ["title", "release_year", "genre_names"] if c in demo_results.columns]
        if not demo_results.empty and cols:
            print(demo_results[cols].head())
        else:
            print("  (No demo results)")

if __name__ == "__main__":
    main()  
