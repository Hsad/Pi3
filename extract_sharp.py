"""Extract sharp (non-blurry) frames from a video using Laplacian variance."""
import argparse
import os
import cv2
import numpy as np

def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("out_dir")
    parser.add_argument("--sample", type=int, default=3,
                        help="Evaluate every Nth frame (default: 3)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Min blur score to keep. Auto-detected if omitted.")
    parser.add_argument("--max_frames", type=int, default=150,
                        help="Max frames to save (default: 150)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f}fps")

    # Pass 1: score every Nth frame
    print(f"Scoring every {args.sample} frames…")
    scored = []  # (frame_idx, score, frame)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.sample == 0:
            scored.append((idx, blur_score(frame), frame))
        idx += 1
    cap.release()

    scores = np.array([s for _, s, _ in scored])
    print(f"Scored {len(scored)} frames — min {scores.min():.1f}, median {np.median(scores):.1f}, max {scores.max():.1f}")

    # Auto threshold: keep frames above 40th percentile (top 60%)
    threshold = args.threshold if args.threshold else float(np.percentile(scores, 40))
    print(f"Blur threshold: {threshold:.1f}")

    sharp = [(i, s, f) for i, s, f in scored if s >= threshold]
    print(f"Sharp frames: {len(sharp)} / {len(scored)}")

    # Evenly subsample down to max_frames
    if len(sharp) > args.max_frames:
        step = len(sharp) / args.max_frames
        sharp = [sharp[int(i * step)] for i in range(args.max_frames)]
        print(f"Subsampled to {len(sharp)} frames")

    # Save
    for n, (frame_idx, score, frame) in enumerate(sharp):
        path = os.path.join(args.out_dir, f"frame_{n:04d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved {len(sharp)} frames to {args.out_dir}")

if __name__ == "__main__":
    main()
