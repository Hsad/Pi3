#!/usr/bin/env python3
"""Flask server: static files, upload management, Pi3 inference jobs."""
import json
import re
import subprocess
import threading
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, redirect, request, send_file, send_from_directory

ROOT         = Path(__file__).parent.resolve()
EXAMPLES     = ROOT / "examples"
SPLATS       = ROOT / "splats"
UPLOADS      = ROOT / "uploads"
APEX_DATASETS = ROOT / "ApexRuns" / "datasets"
UPLOADS.mkdir(exist_ok=True)
SPLATS.mkdir(exist_ok=True)

app  = Flask(__name__)
jobs: dict[str, dict] = {}

@app.after_request
def add_coi_headers(response):
    response.headers["Cross-Origin-Opener-Policy"]   = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response
_lock = threading.Lock()

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def safe(name: str) -> str:
    """Allow only word chars, dash, dot — no path traversal."""
    return re.sub(r"[^\w.\-]", "_", name)


# ── PLY shuffle helper ────────────────────────────────────────────────────────

def _shuffled_ply_response(path):
    """Stream a PLY file with vertex order shuffled so spatial regions interleave."""
    TYPE_SIZES = {
        'float': 4, 'float32': 4, 'double': 8, 'float64': 8,
        'uchar': 1, 'uint8': 1, 'char': 1, 'int8': 1,
        'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
        'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
    }
    data = path.read_bytes()
    marker = b'end_header\n'
    mi = data.find(marker)
    if mi == -1:
        return send_file(path)
    header_end = mi + len(marker)
    header = data[:header_end]
    body   = data[header_end:]

    vertex_count, bpv = 0, 0
    for line in header.decode('latin1').split('\n'):
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == 'element' and parts[1] == 'vertex':
            vertex_count = int(parts[2])
        if len(parts) >= 3 and parts[0] == 'property' and parts[1] != 'list':
            bpv += TYPE_SIZES.get(parts[1], 4)

    if vertex_count == 0 or bpv == 0 or len(body) < vertex_count * bpv:
        return send_file(path)

    arr = np.frombuffer(body, dtype=np.uint8)[:vertex_count * bpv].reshape(vertex_count, bpv).copy()
    np.random.shuffle(arr)

    def generate():
        yield header
        yield arr.tobytes()

    return Response(generate(), mimetype='application/octet-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Static ────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect("/examples/viewer.html")

@app.route("/upload")
def upload_page():
    return send_file(ROOT / "upload.html")

@app.route("/splats")
def splat_viewer():
    return send_file(ROOT / "splat_viewer.html")

@app.route("/mobile")
def mobile_viewer():
    return send_file(ROOT / "mobile.html")

@app.route("/splats/list")
def list_splats():
    return jsonify(sorted(f.name for f in SPLATS.glob("*.ply")))

@app.route("/splats/<path:filename>")
def serve_splat(filename):
    return send_from_directory(SPLATS, filename)

@app.route("/examples/models.json")
def models_json():
    plys = sorted(f.name for f in EXAMPLES.glob("*.ply"))
    return jsonify(plys)

@app.route("/examples/<path:filename>")
def serve_example(filename):
    if request.args.get('shuffle') and filename.lower().endswith('.ply'):
        return _shuffled_ply_response(EXAMPLES / filename)
    return send_from_directory(EXAMPLES, filename)


# ── Apex Runs ─────────────────────────────────────────────────────────────────

@app.route("/apex-runs/list")
def list_apex_runs():
    names = sorted(
        d.name for d in APEX_DATASETS.iterdir()
        if d.is_dir() and (d / "pointcloud.ply").exists()
    )
    return jsonify(names)

@app.route("/apex-runs/<name>/pointcloud.ply")
def serve_apex_ply(name):
    safe_name = re.sub(r"[^\w.\-+]", "_", name)
    path = APEX_DATASETS / safe_name / "pointcloud.ply"
    if request.args.get('shuffle'):
        return _shuffled_ply_response(path)
    return send_from_directory(APEX_DATASETS / safe_name, "pointcloud.ply")


# ── Folders ───────────────────────────────────────────────────────────────────

@app.route("/folders", methods=["GET"])
def list_folders():
    out = []
    for d in sorted(UPLOADS.iterdir()):
        if not d.is_dir():
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        out.append({
            "name":      d.name,
            "count":     len(files),
            "has_video": any(f.suffix.lower() in VIDEO_EXTS for f in files),
        })
    return jsonify(out)

@app.route("/folders/<folder>", methods=["POST"])
def create_folder(folder):
    (UPLOADS / safe(folder)).mkdir(parents=True, exist_ok=True)
    return jsonify({"ok": True})

@app.route("/folders/<folder>", methods=["DELETE"])
def delete_folder(folder):
    import shutil
    p = UPLOADS / safe(folder)
    if p.exists():
        shutil.rmtree(p)
    return jsonify({"ok": True})

@app.route("/folders/<folder>/extract-video", methods=["POST"])
def extract_video(folder):
    import cv2
    data     = request.json or {}
    interval = max(1, int(data.get("interval", 5)))
    d        = UPLOADS / safe(folder)
    videos   = [f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTS]
    if not videos:
        return jsonify({"error": "No video found in this set"}), 404

    video_path = videos[0]
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            cv2.imwrite(str(d / f"frame_{saved:04d}.jpg"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return jsonify({"extracted": saved, "video": video_path.name})


@app.route("/folders/<folder>/files", methods=["GET"])
def list_files(folder):
    d = UPLOADS / safe(folder)
    if not d.exists():
        return jsonify([])
    return jsonify(sorted(f.name for f in d.iterdir() if f.is_file()))

@app.route("/folders/<folder>/files", methods=["POST"])
def upload_files(folder):
    d = UPLOADS / safe(folder)
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in request.files.getlist("files"):
        name = safe(Path(f.filename).name)
        f.save(d / name)
        saved.append(name)
    return jsonify({"saved": saved})

@app.route("/folders/<folder>/files/<filename>", methods=["DELETE"])
def delete_file(folder, filename):
    p = UPLOADS / safe(folder) / safe(filename)
    if p.exists():
        p.unlink()
    return jsonify({"ok": True})


# ── Generate ──────────────────────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
def generate():
    data        = request.json or {}
    folder_name = safe(data.get("folder", ""))
    if not folder_name:
        return jsonify({"error": "No folder"}), 400
    folder = UPLOADS / folder_name
    if not folder.exists():
        return jsonify({"error": "Folder not found"}), 404

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    videos = [f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTS]

    if images:
        # Images always take priority — use the directory, all images included (interval handled client-side as 1)
        data_path = str(folder)
    elif videos:
        # Pure video set — use first video, frame skip applied
        data_path = str(videos[0])
    else:
        return jsonify({"error": "No images or video found in set"}), 400

    interval    = max(1, int(data.get("interval",      10)))
    max_frames  = max(0, int(data.get("max_frames",     0)))
    conf        = float(data.get("conf_threshold",  0.10))
    edge_rtol   = float(data.get("edge_rtol",       0.03))
    voxel_size  = float(data.get("voxel_size",      0.02))
    pixel_limit = int(data.get("pixel_limit",    255000))
    out_name    = safe(data.get("output_name", folder_name)) + ".ply"
    out_path    = str(EXAMPLES / out_name)

    job_id = uuid.uuid4().hex[:8]
    with _lock:
        jobs[job_id] = {"status": "running", "log": [], "output": out_name}

    def run():
        python = str(ROOT / "venv/bin/python")
        cmd = [
            python, str(ROOT / "infer.py"),
            "--data_path",      data_path,
            "--save_path",      out_path,
            "--interval",       str(interval),
            "--max_frames",     str(max_frames),
            "--conf_threshold", str(conf),
            "--edge_rtol",      str(edge_rtol),
            "--voxel_size",     str(voxel_size),
            "--pixel_limit",    str(pixel_limit),
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                with _lock:
                    jobs[job_id]["log"].append(line.rstrip())
            proc.wait()
            status = "done" if proc.returncode == 0 else "error"
        except Exception as exc:
            with _lock:
                jobs[job_id]["log"].append(str(exc))
            status = "error"
        with _lock:
            jobs[job_id]["status"] = status

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/generate_splat", methods=["POST"])
def generate_splat():
    data        = request.json or {}
    folder_name = safe(data.get("folder", ""))
    if not folder_name:
        return jsonify({"error": "No folder"}), 400
    folder = UPLOADS / folder_name
    if not folder.exists():
        return jsonify({"error": "Folder not found"}), 404

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    videos = [f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTS]

    if images:
        data_path = str(folder)
    elif videos:
        data_path = str(videos[0])
    else:
        return jsonify({"error": "No images or video found in set"}), 400

    interval   = max(1, int(data.get("interval",    10)))
    conf       = float(data.get("conf_threshold", 0.10))
    iterations = int(data.get("iterations",       7000))
    out_name   = safe(data.get("output_name", folder_name)) + ".ply"
    out_path   = str(SPLATS / out_name)

    job_id = uuid.uuid4().hex[:8]
    with _lock:
        jobs[job_id] = {"status": "running", "log": [], "output": out_name, "type": "splat"}

    def run():
        python = str(ROOT / "venv/bin/python")
        cmd = [
            python, str(ROOT / "train_splat.py"),
            "--data_path",  data_path,
            "--save_path",  out_path,
            "--interval",   str(interval),
            "--conf",       str(conf),
            "--iterations", str(iterations),
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                with _lock:
                    jobs[job_id]["log"].append(line.rstrip())
            proc.wait()
            status = "done" if proc.returncode == 0 else "error"
        except Exception as exc:
            with _lock:
                jobs[job_id]["log"].append(str(exc))
            status = "error"
        with _lock:
            jobs[job_id]["status"] = status

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/jobs/<job_id>")
def job_stream(job_id):
    import time

    def stream():
        seen = 0
        while True:
            with _lock:
                job    = jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                    return
                lines  = job["log"][seen:]
                seen  += len(lines)
                status = job["status"]
                output = job.get("output")

            for line in lines:
                yield f"data: {json.dumps({'log': line})}\n\n"
            yield f"data: {json.dumps({'status': status, 'output': output})}\n\n"
            if status in ("done", "error"):
                return
            time.sleep(0.35)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9042, threaded=True)
