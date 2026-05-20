#!/usr/bin/env python3
"""
Apex Speed Run point-cloud pipeline.

Courses are grouped by GPS proximity (<=500m) into shared datasets.
Only truly adjacent courses (MILLENNIUM+HIGHLAND+SPEER and AURARIA-1+TIVOLI)
are merged; everything else gets its own dataset.

Priority order for Pi3: datasets with the most contributing videos run first.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

VENV_PYTHON = str(Path(__file__).parent / "venv/bin/python")
VENV_YT_DLP = str(Path(__file__).parent / "venv/bin/yt-dlp")
INFER_PY = str(Path(__file__).parent / "infer.py")
OUTPUT_ROOT = Path(__file__).parent / "ApexRuns"
DATASETS_ROOT = Path(__file__).parent / "ApexRuns" / "datasets"

# (course_slug, city_folder, lat, lon, youtube_url)
COURSES = [
    ("BFI",            "London_England",                  51.5069,   -0.11556,  "https://youtube.com/shorts/Ya53XWYTD7U"),
    ("PASSBY",         "Los-Angeles_California_USA",      33.97461, -118.39257, "https://youtube.com/shorts/A27ETKzL0tQ"),
    ("FUNSTON",        "San-Francisco_California_USA",    37.71516, -122.50444, "https://youtube.com/shorts/JJbYh5Iayeo"),
    ("SHERWOOD",       "San-Francisco_California_USA",    37.80241, -122.46398, "https://www.youtube.com/shorts/0ev-mu_RiIc"),
    ("AUCOIN",         "Portland_Oregon_USA",             45.51053, -122.7171,  "https://youtube.com/shorts/_KvJ6aOqj_U"),
    ("JEFFERSON",      "Portland_Oregon_USA",             45.56059, -122.67286, "https://www.youtube.com/shorts/domOImXkBZg"),
    ("WATER-GARDEN",   "Olympia_Washington_USA",          47.03477, -122.8991,  "https://youtube.com/shorts/okWRNXDmaho"),
    ("IRON-WORKS",     "Vancouver_BC_Canada",             49.28417, -123.09925, "https://www.youtube.com/shorts/uuBlUAXNNJM"),
    ("POINT-GREY",     "Vancouver_BC_Canada",             49.2701,  -123.2594,  "https://youtube.com/shorts/tntUwyNAEOo"),
    ("C4C",            "Boulder_Colorado_USA",            40.00475, -105.26487, "https://www.youtube.com/shorts/Y8IYcoeE09M"),
    ("WATERFRONT",     "Cape-Town_South-Africa",         -33.90415,  18.41766,  "https://youtube.com/shorts/jrLiWp475hQ"),
    ("MINES-1",        "Golden_Colorado_USA",             39.7488,  -105.22318, "https://www.youtube.com/shorts/FPrto9wOBAM"),
    ("EVRY",           "Evry_France",                     48.6245,    2.42752,  "https://youtube.com/shorts/FDwE7I0siao"),
    ("MILLENNIUM",     "Denver_Colorado_USA",             39.75412, -105.00385, "https://www.youtube.com/shorts/b91EJk2nARE"),
    ("CSU",            "Fort-Collins_Colorado_USA",       40.57469, -105.07997, "https://www.youtube.com/shorts/4POMvOf1GBo"),
    ("FESTIVAL",       "Castle-Rock_Colorado_USA",        39.37103, -104.85977, "https://youtube.com/shorts/qc3ZQwkhBPo"),
    ("HIGHLAND",       "Denver_Colorado_USA",             39.75617, -105.00660, "https://www.youtube.com/shorts/jIFqA0OwWv8"),
    ("AURARIA-1",      "Denver_Colorado_USA",             39.74391, -105.0048,  "https://youtube.com/shorts/wbOWUBXk3UI"),
    ("WEST",           "Greenwood-Village_Colorado_USA",  39.62298, -104.92481, "https://www.youtube.com/shorts/llVXOsZv8T0"),
    ("WILL-VILL",      "Boulder_Colorado_USA",            39.99951, -105.25174, "https://youtube.com/shorts/hMLeURn7dj4"),
    ("ACC",            "Littleton_Colorado_USA",          39.60789, -105.01897, "https://youtube.com/shorts/DV-WEGDll3U"),
    ("HARBOURFRONT-2", "Toronto_Ontario_Canada",          43.63913,  -79.37852, "https://youtube.com/shorts/Mh6v7Gptz6Q"),
    ("URBAN-LIFE",     "Atlanta_Georgia_USA",             33.75225,  -84.38596, "https://youtube.com/shorts/0zQ26CYrC5c"),
    ("HANCE-1",        "Phoenix_Arizona_USA",             33.46113, -112.07349, "https://www.youtube.com/shorts/4r-SFdNClI8"),
    ("KAKAAKO",        "Honolulu_Hawaii_USA",             21.29356, -157.86433, "https://www.youtube.com/shorts/Uijo1Kzbino"),
    ("MAKAPUU",        "Waimanalo_Hawaii_USA",            21.30935, -157.65074, "https://youtube.com/shorts/hhbdmwkQs8w"),
    ("UH3",            "Honolulu_Hawaii_USA",             21.30027, -157.81854, "https://www.youtube.com/shorts/bbzcv89CMts"),
    ("ANTIGUA",        "Antigua-Guatemala_Guatemala",     14.56428,  -90.73159, "https://youtube.com/shorts/I8yYRDWjy-A"),
    ("OBELISCO",       "Guatemala-City_Guatemala",        14.59392,  -90.51753, "https://youtube.com/shorts/JVZTqVfG_-c"),
    ("PARCUR",         "Mexico-City_Mexico",              19.39924,  -99.21924, "https://www.youtube.com/shorts/lmk2mjfG-00"),
    ("CUSCATLAN-1",    "San-Salvador_El-Salvador",        13.69937,  -89.20691, "https://www.youtube.com/shorts/z5XciPj48yQ"),
    ("TIVOLI",         "Denver_Colorado_USA",             39.74509, -105.00637, "https://www.youtube.com/shorts/1T2ayz7zAyo"),
    ("HILL",           "Denver_Colorado_USA",             39.72391, -104.93581, "https://www.youtube.com/shorts/CdT0dASobG8"),
    ("TP",             "Lisbon_Portugal",                 38.70453,   -9.16799, "https://www.youtube.com/shorts/FJ1yNyTptCM"),
    ("KAOS",           "Lisbon_Portugal",                 38.70534,   -9.22626, "https://www.youtube.com/shorts/5J_O6Ytj-fo"),
    ("WHALE-TAIL",     "Lisbon_Portugal",                 38.67645,   -9.32072, "https://www.youtube.com/shorts/GM1yOtX5ixo"),
    ("RED-WALLS",      "Almada_Portugal",                 38.66903,   -9.16736, "https://www.youtube.com/shorts/YPyfro4XtYU"),
    ("HORTA-2",        "Lisbon_Portugal",                 38.70773,   -9.2316,  "https://youtube.com/shorts/Q63vWf6HZks"),
    ("GARRISON",       "Arvada_Colorado_USA",             39.801,   -105.10021, "https://www.youtube.com/shorts/t6ODY7r0a_c"),
    ("PCHS",           "Pearl-City_Hawaii_USA",           21.41388, -157.95145, "https://www.youtube.com/shorts/0bfJcZJA2RU"),
    ("UW",             "Seattle_Washington_USA",          47.65079, -122.31135, "https://youtube.com/shorts/4viGhlXuMXs"),
    ("DAIRY",          "Mercer-Island_Washington_USA",    47.59428, -122.22694, "https://www.youtube.com/shorts/HNirKVgwkxk"),
    ("STEAMSHIP",      "Victoria_BC_Canada",              48.42108, -123.37062, "https://youtube.com/shorts/TzH41KtIa7s"),
    ("GAS-WORKS-2",    "Seattle_Washington_USA",          47.64585, -122.33374, "https://www.youtube.com/shorts/nS_o3y2caMk"),
    ("RHODES-1",       "Boise_Idaho_USA",                 43.61799, -116.21325, "https://youtube.com/shorts/0E9Ocgj1i5w"),
    ("UU1",            "Salt-Lake-City_Utah_USA",         40.76086, -111.85127, "https://www.youtube.com/shorts/8EHZBIbOuLs"),
    ("USH",            "Provo_Utah_USA",                  40.23394, -111.63139, "https://youtube.com/shorts/Dl2_RMW1mFY"),
    ("WINCHESTER",     "Las-Vegas_Nevada_USA",            36.13257, -115.10879, "https://youtube.com/shorts/PPxQBBrKoqA"),
    ("UNLV",           "Las-Vegas_Nevada_USA",            36.10566, -115.14238, "https://www.youtube.com/shorts/ekrYuhAvKaI"),
    ("HCH",            "Henderson_Nevada_USA",            36.02912, -114.98264, "https://youtube.com/shorts/YcJAHTfm4oM"),
    ("PCRC",           "Pearl-City_Hawaii_USA",           21.40215, -157.96306, "https://www.youtube.com/shorts/3IaN0vj2U3U"),
    ("STUART",         "Denver_Colorado_USA",             39.71173, -105.04322, "https://www.youtube.com/shorts/w2gd5UhcmgU"),
    ("MCILVOY",        "Arvada_Colorado_USA",             39.80157, -105.07772, "https://youtube.com/shorts/G55bUFl_C2E"),
    ("SPEER",          "Denver_Colorado_USA",             39.7529,  -105.0086,  "https://www.youtube.com/shorts/g5hcB1htJd8"),
    ("BENEFICENCIA",   "San-Juan_Puerto-Rico",            18.46701,  -66.12009, "https://youtube.com/shorts/sZbnBUtXNGs"),
    ("COLISEO",        "Trujillo-Alto_Puerto-Rico",       18.35188,  -66.00615, "https://www.youtube.com/shorts/pBZvYY0mNAo"),
    ("NAVFAC",         "Aguadilla_Puerto-Rico",           18.4885,   -67.16047, "https://www.youtube.com/shorts/Zew2VFAVHAI"),
    ("KAPOLEI",        "Kapolei_Hawaii_USA",              21.32924, -158.0819,  "https://www.youtube.com/shorts/mW19kxzjpVQ"),
    ("JAFFA",          "Tel-Aviv_Israel",                 32.03875,   34.74588, "https://www.youtube.com/shorts/hTRVKimH7dg"),
    ("WAIPIO",         "Waipahu_Hawaii_USA",              21.37443, -158.00028, "https://youtube.com/shorts/EuIS9YofU2o"),
    ("GRAN-CANAL-1",   "Mexico-City_Mexico",              19.43679,  -99.10703, "https://youtube.com/shorts/ydHrhtYZK_M"),
    ("LOMAS-VERDES",   "Naucalpan_Mexico",                19.52012,  -99.27516, "https://youtube.com/shorts/beBY-IKf4ns"),
    ("KAIMUKI-1",      "Honolulu_Hawaii_USA",             21.28253, -157.80056, "https://youtube.com/shorts/YYcqD6vli_k"),
]

import re as _re


def base_name(course_slug):
    """Strip trailing -N suffix: 'AURARIA-1' -> 'AURARIA', 'GAS-WORKS-2' -> 'GAS-WORKS'."""
    return _re.sub(r'-\d+$', '', course_slug)


def build_name_clusters():
    """
    Group courses that share the same base name (e.g. AURARIA-1 + AURARIA-2).
    Each group is a list of course indices.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, (slug, *_) in enumerate(COURSES):
        buckets[base_name(slug)].append(i)
    return list(buckets.values())


def dataset_name(indices):
    """Stable slug for a cluster."""
    names = [COURSES[i][0] for i in sorted(indices)]
    return "+".join(names)


def mp4_path(course_idx):
    c = COURSES[course_idx]
    return OUTPUT_ROOT / c[1] / f"{c[0]}.mp4"


def download_video(url, out_path: Path, label: str):
    if out_path.exists() and out_path.stat().st_size > 10_000:
        print(f"  [skip] {label}")
        return True
    print(f"  [dl] {label}: {url}")
    cmd = [
        VENV_YT_DLP,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist", "-o", str(out_path), url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] failed: {r.stderr[-300:]}")
        return False
    return True


def extract_frames(mp4: Path, frames_dir: Path, prefix: str, fps: float = 4.0):
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob(f"{prefix}_*.jpg"))
    if existing:
        return len(existing)
    out = frames_dir / f"{prefix}_%04d.jpg"
    cmd = ["ffmpeg", "-i", str(mp4), "-vf", f"fps={fps}", "-q:v", "2",
           str(out), "-y", "-loglevel", "warning"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] ffmpeg failed for {prefix}: {r.stderr[-200:]}")
        return 0
    count = len(list(frames_dir.glob(f"{prefix}_*.jpg")))
    return count


def run_pi3(data_path: Path, save_path: Path, label: str):
    ply_file = save_path / "pointcloud.ply"
    if ply_file.exists():
        print(f"  [skip] PLY exists: {label}")
        return True
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"\n[Pi3] {label}  ({data_path})")
    cmd = [
        VENV_PYTHON, INFER_PY,
        "--data_path", str(data_path),
        "--save_path", str(ply_file),
        "--interval", "1",
        "--max_frames", "60",
    ]
    r = subprocess.run(cmd)
    ok = r.returncode == 0
    if not ok:
        print(f"  [WARN] Pi3 failed for {label}")
    return ok


def main():
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    clusters = build_name_clusters()

    print(f"{len(COURSES)} courses → {len(clusters)} name-based datasets")
    multi = [(c, dataset_name(c)) for c in clusters if len(c) > 1]
    if multi:
        print("Merged datasets (same location name):")
        for idxs, name in multi:
            print(f"  {name}: {[COURSES[i][0] for i in idxs]}")

    # --- Phase 1: Download ---
    print("\n" + "="*60)
    print("PHASE 1: Downloading videos")
    print("="*60)
    for course, city, lat, lon, url in COURSES:
        mp4 = OUTPUT_ROOT / city / f"{course}.mp4"
        mp4.parent.mkdir(parents=True, exist_ok=True)
        download_video(url, mp4, f"{city}/{course}")

    # --- Phase 2: Prepare datasets ---
    # Always extract frames via ffmpeg so AV1-encoded videos are handled correctly.
    # OpenCV (used by Pi3) cannot decode AV1; ffmpeg can.
    print("\n" + "="*60)
    print("PHASE 2: Preparing datasets (ffmpeg frame extraction)")
    print("="*60)

    dataset_map = {}  # dataset_name -> {data_path, n_videos}
    for idxs in clusters:
        name = dataset_name(idxs)
        frames_dir = DATASETS_ROOT / name / "frames"
        total = 0
        for i in idxs:
            mp4 = mp4_path(i)
            if mp4.exists():
                cnt = extract_frames(mp4, frames_dir, COURSES[i][0])
                total += cnt
            else:
                print(f"  [MISS] {COURSES[i][0]}: mp4 not found")
        if total == 0:
            print(f"  [SKIP] {name}: no frames extracted")
            continue
        n = len(idxs)
        print(f"  {'[multi]' if n > 1 else '[single]'} {name}: {n} video(s), {total} frames")
        dataset_map[name] = {"data_path": str(frames_dir), "n_videos": n}

    # Save plan
    with open(DATASETS_ROOT / "plan.json", "w") as f:
        json.dump(dataset_map, f, indent=2)

    # --- Phase 3: Run Pi3, most videos first ---
    print("\n" + "="*60)
    print("PHASE 3: Running Pi3 (most videos first)")
    print("="*60)

    ordered = sorted(dataset_map.items(), key=lambda kv: -kv[1]["n_videos"])
    print("Order:")
    for name, info in ordered:
        print(f"  [{info['n_videos']}v] {name}")

    success, failed = 0, []
    for name, info in ordered:
        data_path = Path(info["data_path"])
        save_path = DATASETS_ROOT / name
        if not data_path.exists():
            print(f"  [SKIP] {name}: data not found")
            failed.append(name)
            continue
        if run_pi3(data_path, save_path, name):
            success += 1
        else:
            failed.append(name)

    print("\n" + "="*60)
    print(f"DONE: {success}/{len(dataset_map)} succeeded")
    if failed:
        print(f"Failed/skipped: {', '.join(failed)}")
    print("="*60)


if __name__ == "__main__":
    main()
