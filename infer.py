#!/usr/bin/env python3
"""Pi3 inference with fully-exposed parameters, called by the web server."""
import argparse
import os
import sys

# Reduce allocator fragmentation — large N causes many repeated alloc/free cycles
# that leave the pool fragmented. expandable_segments lets CUDA grow the pool in
# place instead of reserving a fixed slab that fragments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def voxel_downsample(points, colors, normals, voxel_size):
    coords = np.floor(points / voxel_size).astype(np.int64)
    mn = coords.min(axis=0)
    coords -= mn
    mx = coords.max(axis=0) + 1
    keys = coords[:, 0] * (mx[1] * mx[2]) + coords[:, 1] * mx[2] + coords[:, 2]
    order = np.argsort(keys)
    _, first = np.unique(keys[order], return_index=True)
    idx = order[first]
    return points[idx], colors[idx], normals[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--save_path",       required=True)
    parser.add_argument("--interval",        type=int,   default=10)
    parser.add_argument("--max_frames",      type=int,   default=0)   # 0 = no limit
    parser.add_argument("--conf_threshold",  type=float, default=0.10)
    parser.add_argument("--edge_rtol",       type=float, default=0.03)
    parser.add_argument("--voxel_size",      type=float, default=0.02)
    parser.add_argument("--pixel_limit",     type=int,   default=255000)
    args = parser.parse_args()

    # stdout is line-buffered so the Flask SSE stream sees lines immediately
    sys.stdout.reconfigure(line_buffering=True)

    from pi3.utils.basic import load_multimodal_data
    from pi3.utils.geometry import depth_normal_edge
    from pi3.models.pi3x import Pi3X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    no_cond = dict(intrinsics=None, poses=None, depths=None)

    imgs, conditions = load_multimodal_data(
        args.data_path, no_cond,
        interval=args.interval,
        PIXEL_LIMIT=args.pixel_limit,
        device=device,
    )

    n_frames = imgs.shape[1]
    if args.max_frames > 0 and n_frames > args.max_frames:
        step = n_frames / args.max_frames
        keep = [round(i * step) for i in range(args.max_frames)]
        imgs = imgs[:, keep]
        n_frames = imgs.shape[1]
        print(f"Subsampled to {n_frames} frames (max_frames={args.max_frames})")
    print(f"Frames loaded: {n_frames}")
    if n_frames == 0:
        print("ERROR: no frames could be loaded.", file=sys.stderr)
        sys.exit(1)

    torch.cuda.empty_cache()
    print("Loading model…")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to(device)
    model.disable_multimodal()

    # Chunk the DINOv2 encoder so peak VRAM is chunk_size frames, not all N.
    # model.encode() calls self.encoder(imgs_flat, is_training=True) once; we
    # patch encode() to do that call in slices then hand the full hidden back.
    import types

    def _encode_chunked(self, imgs_5d, **kw):
        B, N, C, H, W = imgs_5d.shape
        imgs_flat = imgs_5d.reshape(B * N, C, H, W)
        chunk = 32
        parts = []
        for i in range(0, B * N, chunk):
            h = self.encoder(imgs_flat[i:i + chunk], is_training=True)["x_norm_patchtokens"]
            parts.append(h)
            torch.cuda.empty_cache()
        hidden = torch.cat(parts, dim=0)
        # multimodal is disabled; encode() would just return hidden + Nones
        return hidden, None, None, None, None

    model.encode = types.MethodType(_encode_chunked, model)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.empty_cache()
    print("Running inference…")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        res = model(imgs=imgs)

    print(f"Filtering  conf > {args.conf_threshold}  |  edge_rtol = {args.edge_rtol}…")
    masks = torch.sigmoid(res["conf"][..., 0]) > args.conf_threshold
    non_edge = ~depth_normal_edge(res["local_points"], rtol=args.edge_rtol, mask=masks)
    masks = torch.logical_and(masks, non_edge)[0]

    pts  = res["points"][0][masks].cpu().float().numpy()
    cols = (imgs[0].permute(0, 2, 3, 1)[masks].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    nrm  = torch.nn.functional.normalize(
               res["local_points"][0][masks], dim=-1
           ).cpu().float().numpy()

    print(f"Points before downsample: {len(pts):,}")
    if args.voxel_size > 0 and len(pts) > 0:
        pts, cols, nrm = voxel_downsample(pts, cols, nrm, args.voxel_size)
        print(f"Points after downsample:  {len(pts):,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    verts = np.empty(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts["x"],  verts["y"],  verts["z"]  = pts[:, 0],  pts[:, 1],  pts[:, 2]
    verts["nx"], verts["ny"], verts["nz"] = nrm[:, 0],  nrm[:, 1],  nrm[:, 2]
    verts["red"], verts["green"], verts["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], byte_order="<").write(args.save_path)
    print(f"Saved → {args.save_path}")


if __name__ == "__main__":
    main()
