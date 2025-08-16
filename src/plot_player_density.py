from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_player_densities_from_dataframe(
    player_df: pd.DataFrame,
    grid,                      # either (X, Y) tuple OR model.grid (Grid)
    frame_id: int,
    *,
    orientation_bias_deg: float = 0.0,   # fallback if a row doesn't store a bias
    contour_levels: int = 20,
    contour_alpha: float = 0.30,
    off_cmap: str = "Reds",
    on_cmap: str = "Blues",
    arrow_scale: float = 2.0,
    arrow_head_width: float = 0.375,
    arrow_head_length: float = 0.75,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    grid_alpha: float = 0.10,
    title: str = "Individual Player Influence",
    show: bool = True,
):
    """
    Plot precomputed densities for all players in a given frame.

    Expects `player_df` rows with:
      - frameId, x, y, direction, speed
      - density: 2D numpy array from compute_player_densities_dataframe
      - is_off: offense flag (1 = offense, 0 = defense) → color-coded
      - jerseyNumber, displayName (optional for annotation)
      - orientation_bias_deg (optional; falls back to kwarg if absent)

    Parameters
    ----------
    player_df : DataFrame
        Output rows from `compute_player_densities_dataframe`.
    grid : tuple or Grid
        Either (X, Y) meshgrid tuple or `model.grid` object with X, Y attributes.
    frame_id : int
        Which frame to plot.
    orientation_bias_deg : float
        Used if a row lacks an `orientation_bias_deg`.
    contour_levels : int
        Levels for contourf.
    contour_alpha : float
        Alpha for filled contours.
    off_cmap, on_cmap : str
        Colormaps for defense/offense.
    arrow_scale, arrow_head_width, arrow_head_length : float
        Arrow rendering parameters.
    xlim, ylim : tuple, optional
        Plot ranges. Defaults to full field (0–120, 0–53.3).
    grid_alpha : float
        Grid transparency.
    title : str
        Title string.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, ax
    """
    # Accept either (X, Y) or a Grid object
    try:
        X, Y = grid.X, grid.Y
    except AttributeError:
        X, Y = grid

    _MAX_FIELD_X, _MAX_FIELD_Y = 120.0, 53.3

    f = player_df[player_df["frameId"] == frame_id].copy()
    if f.empty:
        print(f"Warning: no players found for frameId={frame_id}")
        return None, None

    aspect = _MAX_FIELD_Y / _MAX_FIELD_X
    fig, ax = plt.subplots(figsize=(20, 20 * aspect))

    # Light yard lines
    for x in range(10, int(_MAX_FIELD_X), 10):
        ax.axvline(x, color="k", linestyle="-", alpha=0.05)

    def theta_from_tracking(deg: float, bias_deg: float) -> float:
        # Model convention → matplotlib radians
        return np.deg2rad((90.0 - deg + bias_deg) % 360.0)

    for _, row in f.iterrows():
        if str(row.get("displayName", "")).lower() == "football":
            continue

        is_off = row.get("is_off", None)
        player_color = "blue" if is_off == 1 else "red"
        density_cmap = on_cmap if is_off == 1 else off_cmap

        # Per-row stored bias if present, otherwise fallback
        row_bias = float(row.get("orientation_bias_deg", orientation_bias_deg))

        Z = row["density"]
        if not isinstance(Z, np.ndarray):
            continue  # skip malformed rows

        Z_masked = np.where(Z > 0.01, Z, np.nan)
        ax.contourf(X, Y, Z_masked, cmap=density_cmap,
                    levels=contour_levels, alpha=contour_alpha)

        x0, y0 = float(row["x"]), float(row["y"])
        ax.scatter(x0, y0, color=player_color, s=200,
                   edgecolor="black", zorder=5)

        # Jersey number
        jn = row.get("jerseyNumber", None)
        if jn is not None and not (isinstance(jn, float) and np.isnan(jn)):
            ax.text(x0, y0, f"{int(jn)}", fontsize=8,
                    ha="center", color="white", zorder=6)

        # Name
        name = row.get("displayName", None)
        if name:
            ax.text(x0, y0 - 1.5, str(name), fontsize=8,
                    ha="center", color="black", zorder=6)

        # Arrow
        dir_deg = row.get("direction", row.get("orientation", None))
        if dir_deg is not None:
            th = theta_from_tracking(float(dir_deg), row_bias)
            spd = float(row.get("speed", 0.0))
            L = arrow_scale * (1.0 + min(max(spd, 0.0), 11.3) / 11.3)
            ax.arrow(
                x0, y0,
                L * np.cos(th), L * np.sin(th),
                head_width=arrow_head_width,
                head_length=arrow_head_length,
                fc="black", ec="black", zorder=4
            )

    ax.set_title(f"{title} — Frame {frame_id}")
    ax.set_xlabel("X (yards)")
    ax.set_ylabel("Y (yards)")
    ax.grid(alpha=grid_alpha)

    ax.set_xlim(*xlim) if xlim else ax.set_xlim(0, _MAX_FIELD_X)
    ax.set_ylim(*ylim) if ylim else ax.set_ylim(0, _MAX_FIELD_Y)

    if show:
        plt.show()

    return fig, ax


def plot_team_densities_small_multiples(
    player_df: pd.DataFrame,
    grid,                                  # (X, Y) tuple OR model.grid (Grid with .X/.Y)
    *,
    # which frames to show
    frames: Optional[Sequence[int]] = None,
    n_panels: int = 6,                      # if frames=None, pick this many evenly spaced frames
    cols: int = 3,                          # layout columns
    # appearance
    contour_levels: int = 18,
    contour_alpha: float = 0.30,
    off_cmap: str = "Reds",
    on_cmap: str  = "Blues",
    density_threshold: float = 0.01,        # mask tiny values to reduce speckle
    draw_players: bool = True,
    draw_numbers: bool = True,
    draw_names: bool = False,
    draw_arrows: bool = True,
    draw_ball: bool = True,
    arrow_scale: float = 2.0,
    arrow_head_width: float = 0.5,
    arrow_head_length: float = 1.0,
    # angle/bias
    orientation_bias_deg: float = 0.0,      # fallback if a row lacks 'orientation_bias_deg'
    # view/grid
    field_x_max: float = 120.0,
    field_y_max: float = 53.3,
    grid_alpha: float = 0.12,
    # --- NEW: zoom options ---------------------------------------------------
    # zoom_mode:
    #   "none"  -> full field
    #   "ball"  -> center on ball each frame
    #   "los"   -> center on line of scrimmage (fixed x each frame)
    #   "fixed" -> center on a provided (x, y)
    zoom_mode: str = "none",
    zoom_width: float = 30.0,               # x-range (yards) of the zoom window
    zoom_height: Optional[float] = None,    # y-range; if None, use full field_y_max
    los_x: Optional[float] = None,          # LOS x if zoom_mode="los" and no per-frame column
    los_x_col: str = "los_x",               # optional per-frame LOS x column in player_df
    fixed_center: Optional[Tuple[float, float]] = None,  # center if zoom_mode="fixed"
    # title / show
    title_prefix: str = "Team Influence (Gaussian–Gamma) • small multiples",
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray, List[int]]:
    """
    Small-multiples of **full-team** influence for multiple frames, with optional zoom.

    Expects `player_df` produced by `compute_player_densities_dataframe` with rows:
      - frameId, nflId, x, y, speed, direction, is_off
      - density (2D np.ndarray per row)
      - orientation_bias_deg (optional; falls back to kwarg)

    Zoom modes
    ----------
    - "none": full-field view.
    - "ball": per-frame center at the ball; falls back to full field if ball not found.
    - "los": per-frame center at LOS x; uses `los_x_col` if present on any row of the frame,
             else the `los_x` argument (constant across frames). y is centered mid-field.
    - "fixed": use `fixed_center=(x, y)` for all frames.

    Returns
    -------
    fig : Figure
    axes : np.ndarray (flattened)
    frames_used : list[int]
        The exact frame IDs rendered.
    """
    # Unpack grid
    try:
        X, Y = grid.X, grid.Y
    except AttributeError:
        X, Y = grid

    # Which frames to plot
    unique_frames = np.sort(player_df["frameId"].unique())
    if unique_frames.size == 0:
        raise ValueError("player_df has no frameId values.")

    if frames is None:
        # pick evenly spaced frames, then snap to nearest available (avoid duplicates)
        target = np.linspace(unique_frames.min(), unique_frames.max(), max(1, n_panels))
        frames_used: List[int] = []
        for t in target:
            nearest = int(unique_frames[np.argmin(np.abs(unique_frames - t))])
            if not frames_used or frames_used[-1] != nearest:
                frames_used.append(nearest)
    else:
        frames_used = [int(f) for f in frames if f in set(unique_frames)]
        if not frames_used:
            raise ValueError("None of the requested frames are present in player_df.")

    # Layout
    n = len(frames_used)
    rows = int(np.ceil(n / max(1, cols)))
    aspect = field_y_max / field_x_max
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.8 * rows * aspect))
    axes = np.array(axes).reshape(-1)

    # Angle converter (consistent with your model)
    def theta_from_tracking(deg: float, bias_deg: float) -> float:
        # 0° = +Y (north), clockwise → matplotlib (0 rad = +X, CCW)
        return np.deg2rad((90.0 - float(deg) + float(bias_deg)) % 360.0)

    # Helpers for zoom centers
    def _ball_center(f: pd.DataFrame) -> Optional[Tuple[float, float]]:
        # try 'team'==football else displayName=='football'
        if "team" in f.columns:
            b = f[f["team"].str.lower() == "football"] if f["team"].dtype == object else pd.DataFrame()
            if not b.empty:
                return float(b["x"].iloc[0]), float(b["y"].iloc[0])
        if "displayName" in f.columns:
            b = f[f["displayName"].str.lower() == "football"]
            if not b.empty:
                return float(b["x"].iloc[0]), float(b["y"].iloc[0])
        return None

    def _los_center(f: pd.DataFrame) -> Optional[Tuple[float, float]]:
        # look for a per-frame 'los_x' value on any row
        if los_x_col in f.columns and f[los_x_col].notna().any():
            x_val = float(f[los_x_col].dropna().iloc[0])
            return x_val, field_y_max / 2.0
        if los_x is not None:
            return float(los_x), field_y_max / 2.0
        return None

    # Compute per-panel
    for ax, fid in zip(axes, frames_used):
        f = player_df[player_df["frameId"] == fid]
        if f.empty:
            ax.axis("off")
            continue

        # Yard lines
        ax.set_xlim(0, field_x_max)
        ax.set_ylim(0, field_y_max)
        for x in range(10, int(field_x_max), 10):
            ax.axvline(x, color="k", lw=1, alpha=0.06)

        # Densities (overlay all players)
        for _, row in f.iterrows():
            if str(row.get("displayName", "")).lower() == "football":
                continue
            Z = row.get("density", None)
            if Z is None or not isinstance(Z, np.ndarray):
                continue

            is_off = row.get("is_off", None)
            cmap = on_cmap if is_off == 1 else off_cmap

            Zm = np.where(Z > density_threshold, Z, np.nan)
            ax.contourf(X, Y, Zm, levels=contour_levels, cmap=cmap, alpha=contour_alpha)

        # Player dots, numbers, names, arrows
        for _, row in f.iterrows():
            name_lower = str(row.get("displayName", "")).lower()
            is_ball = (name_lower == "football")
            x0, y0 = float(row["x"]), float(row["y"])

            # Ball marker (on top so it stays visible)
            if draw_ball and is_ball:
                ax.scatter(x0, y0, s=120, c="black", edgecolor="white", linewidth=0.6, zorder=8)

            if name_lower == "football":
                continue  # skip other styling for the ball row

            is_off = row.get("is_off", None)
            dot_color = "blue" if is_off == 1 else "red"

            if draw_players:
                ax.scatter(x0, y0, color=dot_color, s=120, edgecolor="black", zorder=5)

            if draw_numbers:
                jn = row.get("jerseyNumber", None)
                if jn is not None and not (isinstance(jn, float) and np.isnan(jn)):
                    ax.text(x0, y0, f"{int(jn)}", fontsize=8,
                            ha="center", color="white", zorder=6)

            if draw_names:
                name = row.get("displayName", None)
                if name:
                    ax.text(x0, y0 - 1.4, str(name), fontsize=8,
                            ha="center", color="black", zorder=6)

            if draw_arrows:
                dir_deg = row.get("direction", None)
                if dir_deg is not None and not (isinstance(dir_deg, float) and np.isnan(dir_deg)):
                    bias = float(row.get("orientation_bias_deg", orientation_bias_deg))
                    th = theta_from_tracking(float(dir_deg), bias)
                    spd = float(row.get("speed", 0.0))
                    L = arrow_scale * (1.0 + min(max(spd, 0.0), 11.3) / 11.3)
                    ax.arrow(
                        x0, y0,
                        L * np.cos(th), L * np.sin(th),
                        head_width=arrow_head_width,
                        head_length=arrow_head_length,
                        fc="black", ec="black", zorder=7
                    )

        # --- Apply zoom AFTER plotting so contour autoscale doesn't fight us ---
        if zoom_mode != "none":
            # default zoom sizes
            zx = float(zoom_width)
            zy = float(zoom_height if zoom_height is not None else field_y_max)

            center: Optional[Tuple[float, float]] = None
            if zoom_mode == "ball":
                center = _ball_center(f)
            elif zoom_mode == "los":
                center = _los_center(f)
            elif zoom_mode == "fixed":
                center = fixed_center

            if center is not None:
                cx, cy = float(center[0]), float(center[1])
                # compute limits and clamp to field bounds
                x_min = max(0.0, cx - zx / 2.0)
                x_max = min(field_x_max, cx + zx / 2.0)
                if x_max - x_min < zx:  # pad if we hit a boundary
                    if x_min == 0.0:
                        x_max = min(field_x_max, x_min + zx)
                    else:
                        x_min = max(0.0, x_max - zx)

                y_min = max(0.0, cy - zy / 2.0)
                y_max = min(field_y_max, cy + zy / 2.0)
                if y_max - y_min < zy:
                    if y_min == 0.0:
                        y_max = min(field_y_max, y_min + zy)
                    else:
                        y_min = max(0.0, y_max - zy)

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

        ax.set_title(f"Frame {fid}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=grid_alpha)

    # Turn off extras
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if show:
        plt.show()

    return fig, axes, frames_used

