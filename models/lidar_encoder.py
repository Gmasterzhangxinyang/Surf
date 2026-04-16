"""LiDAR branch: PointPillars encoder (no spconv needed, pure PyTorch)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarFeatureNet(nn.Module):
    """PointNet-like feature extraction per pillar.

    For each pillar, compute per-point features and aggregate with max pooling.
    """

    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        # Input: (x, y, z, intensity, ring, x_c, y_c, z_c, x_p, y_p)
        # x_c/y_c/z_c = offset from pillar center; x_p/y_p = pillar center coords
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False),
        )
        self.out_channels = out_channels

    def forward(self, pillar_features, pillar_mask):
        """
        Args:
            pillar_features: (N_pillars, max_points, in_channels)
            pillar_mask: (N_pillars, max_points) bool — which points are real

        Returns:
            pillar_out: (N_pillars, out_channels) — max-pooled feature per pillar
        """
        N, P, C = pillar_features.shape

        x = pillar_features.reshape(N * P, C)
        x = self.net(x)
        x = x.reshape(N, P, self.out_channels)

        # Mask out padded points before max pool (non-inplace)
        mask_expanded = pillar_mask.unsqueeze(-1).expand_as(x)
        x = torch.where(mask_expanded, x, torch.tensor(-1e9, device=x.device))
        pillar_out = x.max(dim=1).values  # (N_pillars, out_channels)
        return pillar_out


class PointPillarsScatter(nn.Module):
    """Scatter pillar features onto a 2D BEV pseudo-image."""

    def __init__(self, channels, grid_size):
        super().__init__()
        self.channels = channels
        self.nx, self.ny = grid_size

    def forward(self, pillar_features, pillar_coords):
        """
        Args:
            pillar_features: (N_pillars, C)
            pillar_coords: (N_pillars, 2) — (ix, iy) grid indices

        Returns:
            bev: (1, C, ny, nx) pseudo-image
        """
        C = self.channels
        bev = torch.zeros(C, self.ny * self.nx, device=pillar_features.device)

        ix = pillar_coords[:, 0].long()
        iy = pillar_coords[:, 1].long()

        # Scatter features to BEV grid (non-inplace)
        linear_idx = iy * self.nx + ix  # (N_pillars,)
        linear_idx = linear_idx.unsqueeze(0).expand(C, -1)  # (C, N_pillars)
        bev = bev.scatter_add(1, linear_idx, pillar_features.T)  # (C, ny*nx)
        bev = bev.reshape(C, self.ny, self.nx)

        return bev.unsqueeze(0)  # (1, C, ny, nx)


class BEVBackbone2D(nn.Module):
    """Simple 2D CNN to process the BEV pseudo-image from LiDAR."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.blocks(x)


class LiDAREncoder(nn.Module):
    """Full LiDAR branch: pillarize -> PillarFeatureNet -> scatter -> 2D backbone."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        grid_size = cfg.pillar_grid_size  # (nx, ny)

        # Point features: x,y,z,intensity,ring (5) + offsets x_c,y_c,z_c (3) + pillar center x_p,y_p (2) = 10
        self.pillar_net = PillarFeatureNet(in_channels=10, out_channels=cfg.lidar_channels)
        self.scatter = PointPillarsScatter(cfg.lidar_channels, grid_size)
        self.backbone = BEVBackbone2D(cfg.lidar_channels, cfg.lidar_channels)

    def pillarize(self, points, mask):
        """Convert raw point cloud to pillars.

        Args:
            points: (B, N_pts, 5) — x, y, z, intensity, ring
            mask: (B, N_pts) bool

        Returns per-batch list of:
            pillar_features: (n_pillars, max_pts_per_pillar, 10)
            pillar_mask: (n_pillars, max_pts_per_pillar) bool
            pillar_coords: (n_pillars, 2) — grid (ix, iy)
        """
        cfg = self.cfg
        pcr = cfg.point_cloud_range
        ps = cfg.pillar_size
        nx, ny = cfg.pillar_grid_size
        max_pp = cfg.max_points_per_pillar
        max_pillars = cfg.max_pillars

        results = []
        B = points.shape[0]

        for b in range(B):
            pts = points[b][mask[b]]  # (n_valid, 5)
            if pts.shape[0] == 0:
                results.append((
                    torch.zeros(1, max_pp, 10, device=points.device),
                    torch.zeros(1, max_pp, dtype=torch.bool, device=points.device),
                    torch.zeros(1, 2, dtype=torch.long, device=points.device),
                ))
                continue

            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

            # Compute grid indices
            ix = ((x - pcr[0]) / ps[0]).long().clamp(0, nx - 1)
            iy = ((y - pcr[1]) / ps[1]).long().clamp(0, ny - 1)

            # Hash pillars: unique (ix, iy) pairs
            pillar_id = iy * nx + ix
            unique_ids, inverse = torch.unique(pillar_id, return_inverse=True)

            n_pillars = min(len(unique_ids), max_pillars)
            pillar_features = torch.zeros(n_pillars, max_pp, 10, device=points.device)
            pillar_mask_out = torch.zeros(n_pillars, max_pp, dtype=torch.bool, device=points.device)
            pillar_coords = torch.zeros(n_pillars, 2, dtype=torch.long, device=points.device)

            for p_idx in range(n_pillars):
                pid = unique_ids[p_idx]
                pt_mask = (pillar_id == pid)
                pt_in_pillar = pts[pt_mask]

                n_pts = min(pt_in_pillar.shape[0], max_pp)
                pt_in_pillar = pt_in_pillar[:n_pts]

                # Pillar center
                center_x = pt_in_pillar[:, 0].mean()
                center_y = pt_in_pillar[:, 1].mean()
                center_z = pt_in_pillar[:, 2].mean()

                # Grid center
                p_ix = pid % nx
                p_iy = pid // nx
                grid_x = pcr[0] + (p_ix.float() + 0.5) * ps[0]
                grid_y = pcr[1] + (p_iy.float() + 0.5) * ps[1]

                # Augmented features: [x,y,z,i,ring, x-cx,y-cy,z-cz, grid_x, grid_y]
                offset_x = pt_in_pillar[:, 0] - center_x
                offset_y = pt_in_pillar[:, 1] - center_y
                offset_z = pt_in_pillar[:, 2] - center_z

                aug = torch.stack([
                    pt_in_pillar[:, 0], pt_in_pillar[:, 1], pt_in_pillar[:, 2],
                    pt_in_pillar[:, 3], pt_in_pillar[:, 4],
                    offset_x, offset_y, offset_z,
                    grid_x.expand(n_pts), grid_y.expand(n_pts),
                ], dim=-1)  # (n_pts, 10)

                pillar_features[p_idx, :n_pts] = aug
                pillar_mask_out[p_idx, :n_pts] = True
                pillar_coords[p_idx] = torch.tensor([p_ix, p_iy], device=points.device)

            results.append((pillar_features, pillar_mask_out, pillar_coords))

        return results

    def forward(self, lidar_points, lidar_mask):
        """
        Args:
            lidar_points: (B, N_pts, 5)
            lidar_mask: (B, N_pts) bool

        Returns:
            lidar_bev: (B, C, bev_H, bev_W)
        """
        B = lidar_points.shape[0]
        device = lidar_points.device
        nx, ny = self.cfg.pillar_grid_size

        pillar_data = self.pillarize(lidar_points, lidar_mask)

        bev_list = []
        for b in range(B):
            pf, pm, pc = pillar_data[b]

            # PillarFeatureNet
            pillar_feat = self.pillar_net(pf, pm)  # (n_pillars, C)

            # Scatter to BEV
            bev = self.scatter(pillar_feat, pc)  # (1, C, ny, nx)
            bev_list.append(bev)

        lidar_bev = torch.cat(bev_list, dim=0)  # (B, C, ny, nx)

        # 2D backbone
        lidar_bev = self.backbone(lidar_bev)
        return lidar_bev
