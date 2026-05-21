#!/usr/bin/env python3
# DEPRECATED: Gaussian splat pipeline is no longer in use. Use infer.py for point clouds.
"""
Pi3 → 3D Gaussian Splat pipeline.

Steps:
  1. Load images / video (same as example_mm.py)
  2. Run Pi3X → camera poses + initial point cloud
  3. Optimise 3D Gaussians with gsplat
  4. Export standard 3DGS .ply viewable in any splat viewer
"""
import argparse, math, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pi3.utils.basic   import load_multimodal_data
from pi3.utils.geometry import depth_normal_edge, recover_intrinsic_from_rays_d
from pi3.models.pi3x   import Pi3X
from gsplat            import rasterization

SH_C0 = 0.28209479177387814   # degree-0 SH normalisation constant


# ── helpers ───────────────────────────────────────────────────────────────────

def rgb_to_sh(rgb):
    """Convert [0,1] RGB to SH DC coefficients."""
    return (rgb - 0.5) / SH_C0

def sh_to_rgb(sh):
    return sh * SH_C0 + 0.5

def export_ply(path, means, scales_log, quats, opacities_logit, sh_dc):
    """Write standard 3DGS .ply (raw parameter space)."""
    N   = len(means)
    xyz = means.detach().cpu().numpy()
    sl  = scales_log.detach().cpu().numpy()
    q   = F.normalize(quats, dim=-1).detach().cpu().numpy()
    o   = opacities_logit.detach().cpu().numpy()
    sh  = sh_dc.detach().cpu().numpy()

    dt = np.dtype([
        ('x','f4'),('y','f4'),('z','f4'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
        ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
        ('opacity','f4'),
        ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
        ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
    ])
    v = np.empty(N, dtype=dt)
    v['x'],v['y'],v['z']   = xyz[:,0],xyz[:,1],xyz[:,2]
    v['nx']=v['ny']=v['nz']= 0
    v['f_dc_0'],v['f_dc_1'],v['f_dc_2'] = sh[:,0],sh[:,1],sh[:,2]
    v['opacity']            = o
    v['scale_0'],v['scale_1'],v['scale_2'] = sl[:,0],sl[:,1],sl[:,2]
    v['rot_0'],v['rot_1'],v['rot_2'],v['rot_3'] = q[:,0],q[:,1],q[:,2],q[:,3]

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    PlyData([PlyElement.describe(v, 'vertex')], byte_order='<').write(path)


# ── densification ─────────────────────────────────────────────────────────────

def densify(means, scales_log, quats, opacities_logit, sh_dc,
            grads, grad_threshold, scene_extent, max_gaussians=300_000):
    """Clone small high-grad Gaussians; split large ones; prune invisible ones."""
    scales = torch.exp(scales_log)
    large  = scales.max(dim=-1).values > scene_extent * 0.01
    mask   = grads >= grad_threshold

    # clone (small)
    clone_mask = mask & ~large
    # split (large)
    split_mask = mask & large

    new_means, new_scales, new_quats, new_opac, new_sh = [], [], [], [], []

    if clone_mask.any():
        new_means.append(means[clone_mask])
        new_scales.append(scales_log[clone_mask])
        new_quats.append(quats[clone_mask])
        new_opac.append(opacities_logit[clone_mask])
        new_sh.append(sh_dc[clone_mask])

    if split_mask.any():
        n   = split_mask.sum()
        s   = scales[split_mask]
        q   = F.normalize(quats[split_mask], dim=-1)
        # sample offsets along dominant axis
        R   = torch.zeros(n, 3, 3, device=means.device)
        R[:,0,0]=1-2*(q[:,2]**2+q[:,3]**2); R[:,0,1]=2*(q[:,1]*q[:,2]-q[:,0]*q[:,3]); R[:,0,2]=2*(q[:,1]*q[:,3]+q[:,0]*q[:,2])
        R[:,1,0]=2*(q[:,1]*q[:,2]+q[:,0]*q[:,3]); R[:,1,1]=1-2*(q[:,1]**2+q[:,3]**2); R[:,1,2]=2*(q[:,2]*q[:,3]-q[:,0]*q[:,1])
        R[:,2,0]=2*(q[:,1]*q[:,3]-q[:,0]*q[:,2]); R[:,2,1]=2*(q[:,2]*q[:,3]+q[:,0]*q[:,1]); R[:,2,2]=1-2*(q[:,1]**2+q[:,2]**2)
        offset = (R * s.unsqueeze(1)).sum(-1) * 0.8  # along primary axis
        for sign in [+1, -1]:
            new_means.append(means[split_mask] + sign * offset)
            new_scales.append(scales_log[split_mask] - math.log(1.6))
            new_quats.append(quats[split_mask])
            new_opac.append(opacities_logit[split_mask])
            new_sh.append(sh_dc[split_mask])

    # prune
    keep = torch.sigmoid(opacities_logit) > 0.005

    all_m = [means[keep]] + new_means
    all_s = [scales_log[keep]] + new_scales
    all_q = [quats[keep]] + new_quats
    all_o = [opacities_logit[keep]] + new_opac
    all_c = [sh_dc[keep]] + new_sh

    means          = torch.cat(all_m)[:max_gaussians]
    scales_log     = torch.cat(all_s)[:max_gaussians]
    quats          = torch.cat(all_q)[:max_gaussians]
    opacities_logit= torch.cat(all_o)[:max_gaussians]
    sh_dc          = torch.cat(all_c)[:max_gaussians]

    return means, scales_log, quats, opacities_logit, sh_dc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",   required=True)
    ap.add_argument("--save_path",   required=True)
    ap.add_argument("--interval",    type=int,   default=-1)
    ap.add_argument("--conf",        type=float, default=0.1)
    ap.add_argument("--iterations",  type=int,   default=7000)
    ap.add_argument("--max_init",    type=int,   default=150_000,
                    help="Max Gaussians to seed from Pi3 point cloud")
    args = ap.parse_args()

    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # ── 1. Load images ────────────────────────────────────────────────────────
    print("Loading images…")
    conditions = dict(intrinsics=None, poses=None, depths=None)
    imgs, conditions = load_multimodal_data(
        args.data_path, conditions, interval=args.interval, device=device
    )
    B, N, _, H, W = imgs.shape
    print(f"  {N} frames @ {H}×{W}")

    # ── 2. Run Pi3 ────────────────────────────────────────────────────────────
    print("Running Pi3 inference…")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to(device)
    model.disable_multimodal()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        res = model(imgs=imgs)

    # Camera intrinsics [B, N, 3, 3]
    rays_d = F.normalize(res['local_points'], dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    Ks = K[0].float()                    # [N, 3, 3]

    # Camera poses c2w [N, 4, 4] → w2c for gsplat
    c2w      = res['camera_poses'][0].float()   # [N, 4, 4]
    viewmats = torch.inverse(c2w)               # [N, 4, 4]

    # Initial point cloud
    conf_mask = torch.sigmoid(res['conf'][..., 0]) > args.conf
    non_edge  = ~depth_normal_edge(res['local_points'], rtol=0.03, mask=conf_mask)
    mask      = (conf_mask & non_edge)[0]       # [N, H, W]

    init_xyz  = res['points'][0][mask].float()  # [P, 3]
    init_rgb  = imgs[0].permute(0,2,3,1)[mask].float()  # [P, 3]
    print(f"  Pi3 gave {len(init_xyz):,} seed points")

    del res, model
    torch.cuda.empty_cache()

    # Subsample if too many
    if len(init_xyz) > args.max_init:
        idx      = torch.randperm(len(init_xyz))[:args.max_init]
        init_xyz = init_xyz[idx]
        init_rgb = init_rgb[idx]
        print(f"  Subsampled to {args.max_init:,} Gaussians")

    # ── 3. Initialise Gaussians ───────────────────────────────────────────────
    P = len(init_xyz)
    scene_extent = (init_xyz.max(0).values - init_xyz.min(0).values).max().item()
    init_scale   = math.log(scene_extent / math.sqrt(P) * 0.5)

    means           = init_xyz.clone().requires_grad_(True)
    scales_log      = torch.full((P, 3), init_scale, device=device).requires_grad_(True)
    quats           = torch.zeros(P, 4, device=device); quats[:,0] = 1.0
    quats           = quats.requires_grad_(True)
    opacities_logit = torch.full((P,), -2.0, device=device).requires_grad_(True)
    sh_dc           = rgb_to_sh(init_rgb.clamp(0,1)).requires_grad_(True)

    # Ground-truth images [N, H, W, 3]
    gt = imgs[0].permute(0,2,3,1).float()

    opt = torch.optim.Adam([
        {'params': [means],           'lr': 1.6e-4},
        {'params': [scales_log],      'lr': 5e-3},
        {'params': [quats],           'lr': 1e-3},
        {'params': [opacities_logit], 'lr': 5e-2},
        {'params': [sh_dc],           'lr': 2.5e-3},
    ], eps=1e-15)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print(f"Training {args.iterations} iterations with {P:,} Gaussians…")
    grad_accum = torch.zeros(P, device=device)
    densify_from, densify_until, densify_every = 500, 5000, 100
    opacity_reset_every = 3000

    for i in range(1, args.iterations + 1):
        cam = torch.randint(0, N, (1,)).item()

        renders, alphas, info = rasterization(
            means    = means,
            quats    = F.normalize(quats, dim=-1),
            scales   = torch.exp(scales_log),
            opacities= torch.sigmoid(opacities_logit),
            colors   = sh_to_rgb(sh_dc).clamp(0, 1),
            viewmats = viewmats[cam:cam+1],
            Ks       = Ks[cam:cam+1],
            width=W, height=H,
            absgrad  = True,   # needed for densification grads
        )

        loss = F.l1_loss(renders[0], gt[cam])
        opt.zero_grad()
        loss.backward()

        # accumulate 2-D gradient norms for densification
        if densify_from <= i <= densify_until and info.get("means2d") is not None:
            g = info["means2d"].absgrad
            if g is not None:
                # g shape may be [1, P, 2] or [P, 2]
                g = g.squeeze(0) if g.dim() == 3 else g
                norms = g.norm(dim=-1)
                if norms.shape[0] == grad_accum.shape[0]:
                    grad_accum += norms.detach()

        opt.step()

        # densify
        if densify_from <= i <= densify_until and i % densify_every == 0:
            with torch.no_grad():
                means_d, scales_d, quats_d, opac_d, sh_d = densify(
                    means.detach(), scales_log.detach(), quats.detach(),
                    opacities_logit.detach(), sh_dc.detach(),
                    grad_accum / densify_every, 2e-4, scene_extent,
                )
            P_new = len(means_d)
            means           = means_d.requires_grad_(True)
            scales_log      = scales_d.requires_grad_(True)
            quats           = quats_d.requires_grad_(True)
            opacities_logit = opac_d.requires_grad_(True)
            sh_dc           = sh_d.requires_grad_(True)
            grad_accum      = torch.zeros(P_new, device=device)
            opt = torch.optim.Adam([
                {'params': [means],           'lr': 1.6e-4},
                {'params': [scales_log],      'lr': 5e-3},
                {'params': [quats],           'lr': 1e-3},
                {'params': [opacities_logit], 'lr': 5e-2},
                {'params': [sh_dc],           'lr': 2.5e-3},
            ], eps=1e-15)

        # opacity reset
        if i % opacity_reset_every == 0:
            with torch.no_grad():
                opacities_logit.fill_(-4.0)

        if i % 500 == 0 or i == 1:
            print(f"  [{i:>6}/{args.iterations}] loss={loss.item():.4f}  gaussians={len(means):,}")

    # ── 5. Export ─────────────────────────────────────────────────────────────
    print(f"Exporting to {args.save_path}…")
    export_ply(args.save_path, means, scales_log, quats, opacities_logit, sh_dc)
    print(f"Done — {len(means):,} Gaussians saved.")


if __name__ == "__main__":
    main()
