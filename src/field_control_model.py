from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, gamma



# ---------------------------------------------------------------------------
# Grid: small immutable container to carry X/Y meshgrids together
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    X: np.ndarray  # shape (ny, nx)
    Y: np.ndarray  # shape (ny, nx)

    @property
    def x_vals(self) -> np.ndarray:
        return self.X[0, :]

    @property
    def y_vals(self) -> np.ndarray:
        return self.Y[:, 0]


# ---------------------------------------------------------------------------
# PlayerInfluenceModel
#   Gaussian + Gamma mixture on a football field grid
# ---------------------------------------------------------------------------

class PlayerInfluenceModel:
    """
    Player influence density model (Gaussian + Gamma mixture) with tunable controls.

    Low‑speed behavior
    ------------------
    If `speed < low_speed_gaussian_cutoff`, the model **returns the Gaussian only**
    and forces it to be **perfectly isotropic** (no elongation leakage).
    """

    def __init__(
        self,
        grid_x_res: int = 200,
        grid_y_res: int = 100,
        field_x_max: float = 120.0,
        field_y_max: float = 53.3,
        *,
        # Global (keep bias OFF unless intentionally needed)
        orientation_bias_deg: float = 0.0,
        # Gaussian controls
        gaussian_scale_factor: float = 0.7,
        # Gamma controls
        alpha_gamma: float = 11.0,
        beta_min: float = 1.0,
        beta_max: float = 20.0,
        gamma_midpoint: float = 15.0,
        gamma_scale_factor: float = 0.8,
        max_forward_distance: float = 20.0,
        forward_decay_factor: float = 1.0,
        # Angular cone
        angle_limit_min: float = 15.0,
        angle_limit_max: float = 45.0,
        angle_decay_factor: float = 2.0,
        # Mixture weights (speed→weight logistic)
        w_gaussian_min: float = 0.2,
        w_gaussian_max: float = 1.0,
        gaussian_midpoint: float = 4.0,
        gaussian_steepness: float = 2.0,
        # NEW: low‑speed pure Gaussian cutoff (yd/s)
        low_speed_gaussian_cutoff: float = 2.0,
    ) -> None:
        # Grid / field
        self.grid_x_res = int(grid_x_res)
        self.grid_y_res = int(grid_y_res)
        self.field_x_max = float(field_x_max)
        self.field_y_max = float(field_y_max)
        x_vals = np.linspace(0.0, self.field_x_max, self.grid_x_res)
        y_vals = np.linspace(0.0, self.field_y_max, self.grid_y_res)
        X, Y = np.meshgrid(x_vals, y_vals)
        self.grid = Grid(X=X, Y=Y)

        # Hyperparameters
        self.orientation_bias_deg = float(orientation_bias_deg)

        self.gaussian_scale_factor = float(gaussian_scale_factor)

        self.alpha_gamma = float(alpha_gamma)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.gamma_midpoint = float(gamma_midpoint)
        self.gamma_scale_factor = float(gamma_scale_factor)
        self.max_forward_distance = float(max_forward_distance)
        self.forward_decay_factor = float(forward_decay_factor)

        self.angle_limit_min = float(angle_limit_min)
        self.angle_limit_max = float(angle_limit_max)
        self.angle_decay_factor = float(angle_decay_factor)

        self.w_gaussian_min = float(w_gaussian_min)
        self.w_gaussian_max = float(w_gaussian_max)
        self.gaussian_midpoint = float(gaussian_midpoint)
        self.gaussian_steepness = float(gaussian_steepness)

        self.low_speed_gaussian_cutoff = float(low_speed_gaussian_cutoff)

    # -----------------------------------------------------------------------
    # Angle helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _norm_deg(deg: float) -> float:
        return (deg % 360.0 + 360.0) % 360.0

    def theta_from_tracking(self, deg: float, *, apply_bias: bool = False) -> float:
        eff_deg = 90.0 - deg + (self.orientation_bias_deg if apply_bias else 0.0)
        return np.deg2rad(self._norm_deg(eff_deg))

    # -----------------------------------------------------------------------
    # Mixture weights as a function of speed
    # -----------------------------------------------------------------------

    def dynamic_weights(self, speed: float) -> Tuple[float, float]:
        """Logistic map from speed → (wG, wH), where wH = 1 - wG."""
        wG = self.w_gaussian_min + (self.w_gaussian_max - self.w_gaussian_min) / (
            1.0 + np.exp(self.gaussian_steepness * (speed - self.gaussian_midpoint))
        )
        return float(wG), float(1.0 - wG)

    # -----------------------------------------------------------------------
    # Gaussian component (elliptical local influence)
    # -----------------------------------------------------------------------

    @staticmethod
    def _radius_influence(dist_from_ball: float) -> float:
        """Baseline radius vs. distance to ball (caps ~3 yds)."""
        if dist_from_ball <= 18.0:
            return 1.0 + (3.0 / (18.0 ** 2)) * (dist_from_ball ** 2)
        return 3.0

    def _sigma(
        self,
        theta: float,
        speed: float,
        dist_from_ball: float,
        *,
        max_speed: float = 11.3,
    ) -> np.ndarray:
        """
        Rotated 2×2 covariance matrix for the Gaussian.
        - Elongates along heading with speed
        - **NEW**: Below cutoff, force isotropy (no elongation)
        """
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Speed modulation (squared ratio). Kill elongation when slow.
        ratio = 0.0 if speed < self.low_speed_gaussian_cutoff else float((speed ** 2) / (max_speed ** 2))

        base = self._radius_influence(dist_from_ball)
        sx = (base + base * ratio) * self.gaussian_scale_factor
        sy = max((base - base * ratio) * self.gaussian_scale_factor, sx * 0.5)

        S = np.array([[sx, 0.0], [0.0, sy]])
        cov = R @ (S ** 2) @ R.T
        cov += np.eye(2) * 1e-6
        return cov

    @staticmethod
    def _mu(pos_xy: Tuple[float, float], vel_xy: np.ndarray) -> np.ndarray:
        """Gaussian mean = current position + 0.5 * velocity (gentle forward nudge)."""
        return np.array(pos_xy, dtype=float) + 0.5 * vel_xy

    # -----------------------------------------------------------------------
    # Gamma anchor (origin behind player so mode sits near player)
    # -----------------------------------------------------------------------

    def compute_offset(
        self,
        pos_xy: Tuple[float, float],
        direction_deg: float,
        speed: float
    ) -> Tuple[float, float]:
        beta = self.beta_min + (self.beta_max - self.beta_min) / (1.0 + np.exp(1.0 * (speed - 1.0)))
        mode = (8.0 - 1.0) / beta  # mode of Gamma(k≈8, β)
        th = self.theta_from_tracking(direction_deg, apply_bias=False)
        return (pos_xy[0] - mode * np.cos(th), pos_xy[1] - mode * np.sin(th))

    # -----------------------------------------------------------------------
    # Core density (Gaussian + Gamma), normalized and mixed
    # -----------------------------------------------------------------------

    def base_distribution(
        self,
        pos_xy: Tuple[float, float],
        pos_off_xy: Tuple[float, float],
        direction_deg: float,
        speed: float,
        *,
        dist_from_ball: float,
        alpha_gamma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute mixed influence Z = wG*G + wH*H on the model grid.
        **NEW**: if speed < cutoff → return G only (pure circular Gaussian).
        """
        X, Y = self.grid.X, self.grid.Y
        x_off, y_off = pos_off_xy
        th = self.theta_from_tracking(direction_deg, apply_bias=False)

        # Velocity and angular cone width (narrows with speed)
        vel = np.array([np.cos(th) * speed, np.sin(th) * speed], dtype=float)
        ang_lim_deg = self.angle_limit_min + (self.angle_limit_max - self.angle_limit_min) * np.exp(
            -speed / self.angle_decay_factor
        )

        # ----- Gaussian term -----
        cov = self._sigma(th, speed, dist_from_ball)
        mu = self._mu(pos_xy, vel)
        G = multivariate_normal(mean=mu, cov=cov).pdf(np.dstack((X, Y)))
        G /= G.max() if G.max() > 0 else 1.0  # normalize to [0,1]

        # **NEW: low-speed guard — pure Gaussian, isotropic**
        if speed < self.low_speed_gaussian_cutoff:
            return G

        # ----- Gamma term (forward) -----
        beta_dyn = self.beta_min + (self.beta_max - self.beta_min) * (speed / self.gamma_midpoint)
        beta_dyn = np.clip(beta_dyn, self.beta_min, self.beta_max)

        dx, dy = X - x_off, Y - y_off
        d_proj = dx * np.cos(th) + dy * np.sin(th)                      # distance along heading
        d_scaled = d_proj / max(speed * (1.0 + speed / 18.0), 1e-3)     # speed-scaled axis

        a = float(self.alpha_gamma if alpha_gamma is None else alpha_gamma)
        H = gamma.pdf(d_scaled, a=a, scale=1.0 / (beta_dyn * self.gamma_scale_factor))

        # Soft cap fade
        mask_far = d_proj > self.max_forward_distance
        if np.any(mask_far):
            fade = self.forward_decay_factor if self.forward_decay_factor > 0 else 1.0
            H[mask_far] *= np.exp(
                -((d_proj[mask_far] - self.max_forward_distance) ** 2)
                / (2.0 * (speed * fade) ** 2 + 1e-6)
            )

        # Soft angular gate near heading
        ang_to_pt = np.arctan2(dy, dx)
        ang_diff = np.abs((ang_to_pt - th + np.pi) % (2.0 * np.pi) - np.pi)
        H *= 1.0 / (1.0 + np.exp(10.0 * (ang_diff - np.radians(ang_lim_deg))))

        H /= H.max() if H.max() > 0 else 1.0

        # ----- Mix with speed‑dependent weights -----
        wG, wH = self.dynamic_weights(speed)
        Z = wG * G + wH * H
        return Z

    # -----------------------------------------------------------------------
    # Batch per frame
    # -----------------------------------------------------------------------

    def compute_influence(
        self,
        df: pd.DataFrame,
        frame_id: int,
        player_ids: Optional[Iterable[int]] = None,
        *,
        id_col: str = "nflId",
        name_col: str = "displayName",
        x_col: str = "x",
        y_col: str = "y",
        speed_col: str = "s",
        dir_col: str = "dir",
        dist_from_ball_col: str = "dist_from_football",
        density_out_col: str = "density_grid",
    ) -> pd.DataFrame:
        """Compute per‑player influence grids for a single frame."""
        f = df.loc[df["frameId"] == frame_id].copy()
        if player_ids is not None:
            if isinstance(player_ids, (int, float, str)):
                player_ids = [player_ids]
            f = f[f[id_col].isin(player_ids)].copy()
        if name_col in f.columns:
            f = f[f[name_col].str.lower() != "football"].copy()

        required = {id_col, x_col, y_col, speed_col, dir_col, dist_from_ball_col}
        missing = required - set(f.columns)
        if missing:
            raise KeyError(f"Missing required columns in df: {sorted(missing)}")

        out_rows = []
        for _, r in f.iterrows():
            pos = (float(r[x_col]), float(r[y_col]))
            dir_deg = float(r[dir_col])
            spd = float(r[speed_col])
            dball = float(r[dist_from_ball_col])

            pos_off = self.compute_offset(pos, dir_deg, spd)
            Z = self.base_distribution(
                pos_xy=pos,
                pos_off_xy=pos_off,
                direction_deg=dir_deg,
                speed=spd,
                dist_from_ball=dball,
            )
            rec = r.to_dict()
            rec[density_out_col] = Z
            out_rows.append(rec)

        return pd.DataFrame(out_rows)
