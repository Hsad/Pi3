"""Voxel downsample a PLY point cloud using only numpy + plyfile."""
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def voxel_downsample(points, colors, normals, voxel_size):
    coords = np.floor(points / voxel_size).astype(np.int64)
    # pack 3 int64 coords into a single key for fast unique grouping
    mn = coords.min(axis=0)
    coords -= mn
    mx = coords.max(axis=0) + 1
    keys = coords[:, 0] * (mx[1] * mx[2]) + coords[:, 1] * mx[2] + coords[:, 2]
    order = np.argsort(keys)
    keys_sorted = keys[order]
    _, first = np.unique(keys_sorted, return_index=True)
    idx = order[first]
    return points[idx], colors[idx], normals[idx]

def main():
    parser = argparse.ArgumentParser(description="Voxel-downsample a PLY point cloud")
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", help="Output .ply file")
    parser.add_argument("--voxel", type=float, default=0.02,
                        help="Voxel size in scene units (default: 0.02). Larger = fewer points.")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    ply = PlyData.read(args.input)
    v = ply["vertex"]
    points  = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    normals = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    colors  = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    print(f"  {len(points):,} points loaded")

    print(f"Downsampling with voxel size {args.voxel}...")
    pts, col, nrm = voxel_downsample(points, colors, normals, args.voxel)
    print(f"  {len(pts):,} points remaining ({100*len(pts)/len(points):.1f}%)")

    verts = np.empty(len(pts), dtype=[
        ("x","f4"),("y","f4"),("z","f4"),
        ("nx","f4"),("ny","f4"),("nz","f4"),
        ("red","u1"),("green","u1"),("blue","u1"),
    ])
    verts["x"], verts["y"], verts["z"] = pts[:,0], pts[:,1], pts[:,2]
    verts["nx"], verts["ny"], verts["nz"] = nrm[:,0], nrm[:,1], nrm[:,2]
    verts["red"], verts["green"], verts["blue"] = col[:,0], col[:,1], col[:,2]

    PlyData([PlyElement.describe(verts, "vertex")], byte_order="<").write(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
