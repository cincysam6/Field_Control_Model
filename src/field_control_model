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
    """
    Immutable container for the field grid used by the model.

    Attributes
    ----------
    X : np.ndarray
        2D array of x-coordinates, shape (ny, nx).
    Y : np.ndarray
        2D array of y-coordinates, shape (ny, nx).

    Notes
    -----
    - We keep X and Y together so every density is evaluated on a
      consistent coordinate system (no shape mismatches during plotting).
    """
    X: np.ndarray  # shape (ny, nx)
    Y: np.ndarray  # shape (ny, nx)

    @property
    def x_vals(self) -> np.ndarray:
        """1D x coordinate axis (length nx)."""
        return self.X[0, :]

    @property
    def y_vals(self) -> np.ndarray:
        """1D y coordinate axis (length ny)."""
        return self.Y[:, 0]


# ---------------------------------------------------------------------------
# PlayerInfluenceModel
#   Gaussian + Gamma mixture on a football field grid
# ---------------------------------------------------------------------------

class PlayerInfluenceModel:
    """
    Player influence density model (Gaussian + Gamma mixture) with tunable controls.

    Angle conventions
    -----------------
    Tracking data:
      - 0° = due north (up, +Y)
      - degrees increase clockwise (to 360°)

    Math/Matplotlib:
      - 0 rad = +X, CCW is positive
      - canonical conversion used everywhere:
            theta = radians(90 - deg)

    High‑level intuition
    --------------------
    - The **Gaussian** term captures local, elliptical influence around the player,
      elongated along their heading and modestly scaled by speed.
    - The **Gamma** term captures **forward reach**: mass distributed *ahead*
      of the player along the heading, tapered by a forward cap and angular cone.
    - A **speed‑dependent weight** mixes the two: more Gaussian at low speed,
      more Gamma as speed rises.

    Parameters (hyperparameters)
    ----------------------------
    grid_x_res, grid_y_res : int
        Resolution of the evaluation grid along X and Y.
        Higher = smoother but more compute/memory.

    field_x_max, field_y_max : float
        Field limits in yards (default NFL: 120 × 53.3).

    orientation_bias_deg : float
        Global bias (in degrees) added *only if* `apply_bias=True`
        in `theta_from_tracking`. Keep **0.0** to match tracking as-is.

    gaussian_scale_factor : float
        Scales the Gaussian covariance ellipses (both axes). Larger spreads
        out local influence; smaller concentrates it.

    alpha_gamma : float
        Shape parameter (k) for the **Gamma** pdf along the forward axis.
        Higher α → peakier, lower α → flatter.

    beta_min, beta_max : float
        Bounds for the dynamic rate (β) used by the Gamma component. We ramp
        β with speed to adjust reach in a controlled range.

    gamma_midpoint : float
        Speed (yd/s) at which the **β ramp** is centered (used in a simple
        linear clip). Larger midpoint → β grows more slowly with speed.

    gamma_scale_factor : float
        Additional scaling on the Gamma pdf’s scale (1/β). <1.0 compresses,
        >1.0 extends.

    max_forward_distance : float
        Soft cap (yards) after which forward mass fades rapidly (Gaussian fade).

    forward_decay_factor : float
        Controls how quickly the Gamma mass decays beyond the forward cap.
        Larger = slower fade (longer tail); smaller = faster fade.

    angle_limit_min, angle_limit_max : float (degrees)
        Angular cone around the heading used to **soft‑gate** Gamma mass.
        The live cone is interpolated between these based on speed:
            ang_lim = min + (max - min) * exp(-speed / angle_decay_factor)

    angle_decay_factor : float
        Speed scale for narrowing the cone. Smaller → narrows faster with speed.

    w_gaussian_min, w_gaussian_max : float in [0,1]
        Bounds for the Gaussian weight in the speed‑logistic mix:
        - At high speed the Gaussian weight approaches `w_gaussian_min`.
        - At low speed the Gaussian weight approaches `w_gaussian_max`.

    gaussian_midpoint : float
        Logistic midpoint for the speed→weight mapping (yd/s).

    gaussian_steepness : float
        Logistic steepness. Larger → sharper transition from Gaussian→Gamma
        as speed increases.
    """

    def __init__(
        self,
        grid_x_res: int = 200,
        grid_y_res: int = 100,
        field_x_max: float = 120.0,
        field_y_max: float = 53.3,
        *,
        # keep bias OFF unless you intentionally need it
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
    ) -> None:
        # ---------------- Grid / field setup ---------------------------------
        # Build the evaluation meshgrid once (X,Y). All densities share it.
        self.grid_x_res = int(grid_x_res)
        self.grid_y_res = int(grid_y_res)
        self.field_x_max = float(field_x_max)
        self.field_y_max = float(field_y_max)

        x_vals = np.linspace(0.0, self.field_x_max, self.grid_x_res)
        y_vals = np.linspace(0.0, self.field_y_max, self.grid_y_res)
        X, Y = np.meshgrid(x_vals, y_vals)
        self.grid = Grid(X=X, Y=Y)

        # ---------------- Store hyperparameters ------------------------------
        # Orientation bias (only used when explicitly requested)
        self.orientation_bias_deg = float(orientation_bias_deg)

        # Gaussian: ellipse size scaling
        self.gaussian_scale_factor = float(gaussian_scale_factor)

        # Gamma: shape/rate scaling and forward reach
        self.alpha_gamma = float(alpha_gamma)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.gamma_midpoint = float(gamma_midpoint)
        self.gamma_scale_factor = float(gamma_scale_factor)
        self.max_forward_distance = float(max_forward_distance)
        self.forward_decay_factor = float(forward_decay_factor)

        # Angular cone (soft gate)
        self.angle_limit_min = float(angle_limit_min)
        self.angle_limit_max = float(angle_limit_max)
        self.angle_decay_factor = float(angle_decay_factor)

        # Mixture weights: speed→(wG, wH=1-wG)
        self.w_gaussian_min = float(w_gaussian_min)
        self.w_gaussian_max = float(w_gaussian_max)
        self.gaussian_midpoint = float(gaussian_midpoint)
        self.gaussian_steepness = float(gaussian_steepness)

    # -----------------------------------------------------------------------
    # Angle helpers (single source of truth)
    # -----------------------------------------------------------------------

    @staticmethod
    def _norm_deg(deg: float) -> float:
        """Normalize degrees to [0, 360)."""
        return (deg % 360.0 + 360.0) % 360.0

    def theta_from_tracking(self, deg: float, *, apply_bias: bool = False) -> float:
        """
        Convert tracking degrees (0°=north, clockwise) → matplotlib radians (0 rad = +X, CCW).
        Uses the canonical mapping everywhere:
            theta = radians(90 - deg [+ orientation_bias_deg if apply_bias])
        """
        eff_deg = 90.0 - deg + (self.orientation_bias_deg if apply_bias else 0.0)
        return np.deg2rad(self._norm_deg(eff_deg))

    # -----------------------------------------------------------------------
    # Mixture weights as a function of speed
    # -----------------------------------------------------------------------

    def dynamic_weights(self, speed: float) -> Tuple[float, float]:
        """
        Logistic map from speed → (wG, wH), where wH = 1 - wG.

        Intuition:
        - At low speed → higher Gaussian (more local, symmetric mass).
        - At high speed → lower Gaussian (more forward reach via Gamma).
        """
        wG = self.w_gaussian_min + (self.w_gaussian_max - self.w_gaussian_min) / (
            1.0 + np.exp(self.gaussian_steepness * (speed - self.gaussian_midpoint))
        )
        return float(wG), float(1.0 - wG)

    # -----------------------------------------------------------------------
    # Gaussian component (elliptical local influence)
    # -----------------------------------------------------------------------

    @staticmethod
    def _radius_influence(dist_from_ball: float) -> float:
        """
        Baseline radius as a function of distance to ball; gently grows to a cap ~3 yards.

        Rationale:
        - Slightly increase local footprint when far from ball (less clutter near play).
        """
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
        Build a rotated 2×2 covariance matrix for the Gaussian.
        - Elongates along heading (θ) with speed
        - Shrinks across heading with speed (not below 50% of along‑axis)
        """
        # Rotation matrix to align ellipse with heading θ
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Speed ratio (squared) to modulate axial scales
        ratio = float((speed ** 2) / (max_speed ** 2))

        # Baseline radius grows with distance from ball (then caps)
        base = self._radius_influence(dist_from_ball)

        # Along- and cross-axis stddevs (scaled)
        sx = (base + base * ratio) * self.gaussian_scale_factor
        sy = max((base - base * ratio) * self.gaussian_scale_factor, sx * 0.5)

        # Rotate the diagonal covariance into global coordinates
        S = np.array([[sx, 0.0], [0.0, sy]])
        cov = R @ (S ** 2) @ R.T

        # Tiny ridge for numerical stability
        cov += np.eye(2) * 1e-6
        return cov

    @staticmethod
    def _mu(pos_xy: Tuple[float, float], vel_xy: np.ndarray) -> np.ndarray:
        """
        Gaussian mean = current position + 0.5 * velocity.
        Gives a gentle forward nudge to the local mass.
        """
        return np.array(pos_xy, dtype=float) + 0.5 * vel_xy

    # -----------------------------------------------------------------------
    # Gamma anchor (where "behind" the player the Gamma originates)
    # -----------------------------------------------------------------------

    def compute_offset(
        self,
        pos_xy: Tuple[float, float],
        direction_deg: float,
        speed: float
    ) -> Tuple[float, float]:
        """
        Place the Gamma's reference point slightly *behind* the player so its mode
        lands near the actual player position. The offset depends on speed so the
        forward lobe aligns visually across speeds.
        """
        # Dynamic β via a logistic-like curve near low speed to keep mode stable
        beta = self.beta_min + (self.beta_max - self.beta_min) / (1.0 + np.exp(1.0 * (speed - 1.0)))
        # Mode of a Gamma(k, β): (k-1)/β ; here we use k≈8 (empirically stable)
        mode = (8.0 - 1.0) / beta

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
        Compute the mixed influence:

            Z = wG * G + wH * H

        where:
          - G is a normalized 2D Gaussian aligned with heading θ
          - H is a normalized Gamma lobe projected forward along θ, soft‑gated by angle

        Returns
        -------
        Z : np.ndarray of shape (ny, nx)
            Influence on the model grid.
        """
        X, Y = self.grid.X, self.grid.Y
        x_off, y_off = pos_off_xy
        th = self.theta_from_tracking(direction_deg, apply_bias=False)

        # ---------- Velocity and angular cone width (narrows with speed) -----
        vel = np.array([np.cos(th) * speed, np.sin(th) * speed], dtype=float)
        ang_lim_deg = self.angle_limit_min + (self.angle_limit_max - self.angle_limit_min) * np.exp(
            -speed / self.angle_decay_factor
        )

        # ---------- Gaussian term (local, elliptical) ------------------------
        cov = self._sigma(th, speed, dist_from_ball)
        mu = self._mu(pos_xy, vel)
        G = multivariate_normal(mean=mu, cov=cov).pdf(np.dstack((X, Y)))
        G /= G.max() if G.max() > 0 else 1.0  # normalize to [0,1]

        # ---------- Gamma term (forward projection) --------------------------
        # speed→β (clipped) to adjust reach
        beta_dyn = self.beta_min + (self.beta_max - self.beta_min) * (speed / self.gamma_midpoint)
        beta_dyn = np.clip(beta_dyn, self.beta_min, self.beta_max)

        dx, dy = X - x_off, Y - y_off
        d_proj = dx * np.cos(th) + dy * np.sin(th)  # distance along heading
        d_scaled = d_proj / max(speed * (1.0 + speed / 18.0), 1e-3)

        # shape α and rate β=beta_dyn*gamma_scale_factor → scale = 1/β
        a = float(self.alpha_gamma if alpha_gamma is None else alpha_gamma)
        H = gamma.pdf(d_scaled, a=a, scale=1.0 / (beta_dyn * self.gamma_scale_factor))

        # Fade out beyond forward cap (soft Gaussian-like decay in the tail)
        mask_far = d_proj > self.max_forward_distance
        if np.any(mask_far):
            fade = self.forward_decay_factor if self.forward_decay_factor > 0 else 1.0
            H[mask_far] *= np.exp(
                -((d_proj[mask_far] - self.max_forward_distance) ** 2)
                / (2.0 * (speed * fade) ** 2 + 1e-6)
            )

        # Soft angular gate: keep mass near the heading, fade sideways
        ang_to_pt = np.arctan2(dy, dx)
        ang_diff = np.abs((ang_to_pt - th + np.pi) % (2.0 * np.pi) - np.pi)
        H *= 1.0 / (1.0 + np.exp(10.0 * (ang_diff - np.radians(ang_lim_deg))))

        H /= H.max() if H.max() > 0 else 1.0  # normalize to [0,1]

        # ---------- Mix with speed‑dependent weights -------------------------
        wG, wH = self.dynamic_weights(speed)
        Z = wG * G + wH * H
        return Z

    # -----------------------------------------------------------------------
    # Batch per frame: compute grids for one frame (optionally subset players)
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
        """
        Compute per‑player influence grids for a single frame.

        Returns a DataFrame that mirrors the input rows but adds the numpy array
        in column `density_out_col` for each player.

        Raises
        ------
        KeyError
            If required columns are missing.
        """
        # Filter to the requested frame and (optionally) players; drop the ball.
        f = df.loc[df["frameId"] == frame_id].copy()
        if player_ids is not None:
            if isinstance(player_ids, (int, float, str)):
                player_ids = [player_ids]
            f = f[f[id_col].isin(player_ids)].copy()
        if name_col in f.columns:
            f = f[f[name_col].str.lower() != "football"].copy()

        # Ensure required schema is present
        required = {id_col, x_col, y_col, speed_col, dir_col, dist_from_ball_col}
        missing = required - set(f.columns)
        if missing:
            raise KeyError(f"Missing required columns in df: {sorted(missing)}")

        # Compute density per row
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


