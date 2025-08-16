# src/animate_play.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from matplotlib import animation


__all__ = ["AnimatePlayWithDensity"]


class AnimatePlayWithDensity:
    """
    Animate a play using **precomputed per-player density grids**.

    This class draws players, optional density contours for each player at each
    frame, and direction/orientation arrows that follow your tracking convention:
      - Tracking degrees: 0° = north (+Y), increasing clockwise to 360°
      - Arrow mapping to matplotlib: θ = radians(90 - deg + bias)

    Notes
    -----
    - To avoid lingering “ghost” dots from previous frames, player PathCollections
      are updated **in-place** with *safe* offset/size resets.
    - This animator expects a *precomputed* DataFrame that already contains a 2D
      density array per player & frame (e.g., computed by your
      `compute_player_densities_dataframe`).

    Required schema
    ----------------
    `play_df` (all frames for a play):
      - frameId, x, y, is_off (1=offense, 0=defense), jerseyNumber, displayName
      - dir (player heading in tracking degrees), s (speed)
      - team or displayName=='football' for the ball row

    `precomputed_df` (one row per player per frame):
      - frameId, nflId, x, y, is_off, density (np.ndarray [ny, nx])
      - OPTIONAL: theta_rad (float) resolved direction in radians at compute time
      - OPTIONAL: orientation_bias_deg (float) used during compute

    Parameters
    ----------
    play_df : pd.DataFrame
        Tracking rows for a single play (all frames).
    precomputed_df : pd.DataFrame
        Output from a density precompute step; must include a 'density' array per row.
    grid : tuple(ndarray, ndarray) or object with .X/.Y
        Field mesh (X, Y). Either a (X, Y) tuple or a Grid-like object with `X` and `Y`.
    plot_size_len : float, default 16.0
        Base figure width in inches. Height is set by field aspect (~53.3/120).
    show_contours : bool, default True
        If True, plots the per-player density as filled contours at each frame.
    contour_levels : int, default 20
        Number of contour levels for density plots.
    contour_alpha : float, default 0.30
        Transparency of density contours.
    off_cmap : str, default "Reds"
        Colormap for defenders’ densities.
    on_cmap : str, default "Blues"
        Colormap for offensive players’ densities.
    arrow_scale : float, default 2.0
        Base arrow scale (boosted by `arrow_length_boost` and speed).
    arrow_head_width : float, default 0.50
        Arrow head width (matplotlib Arrow patch).
    arrow_head_length : float, default 1.00
        Arrow head length (matplotlib Arrow patch).
    arrow_length_boost : float, default 1.35
        Extra multiplicative factor to make arrows more readable.
    arrow_tail_width : float, default 0.20
        Shaft width for the Arrow patch (helps visibility).
    orientation_bias_deg : float, default 0.0
        Fallback bias (deg) added to arrows only if a row lacks stored bias.
    xlim, ylim : Optional[Tuple[float, float]]
        Optional axis limits. If omitted, full field (0–120, 0–53.3) is shown.
    grid_alpha : float, default 0.10
        Background grid transparency.
    title : str, default "Play Animation with Influence"
        Title prefix; frame number is appended during the animation.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The underlying Matplotlib figure (`.fig` alias).
    ax : matplotlib.axes.Axes
        The main axis (`.ax` alias).
    ani : matplotlib.animation.FuncAnimation
        The animation object you can render via `.to_jshtml()` or `.save(...)`.

    Example
    -------
    >>> animator = AnimatePlayWithDensity(
    ...     play_df=tracking_play_df,
    ...     precomputed_df=all_player_df,   # from compute_player_densities_dataframe(...)
    ...     grid=(X, Y),
    ...     show_contours=True
    ... )
    >>> from IPython.display import HTML
    >>> HTML(animator.ani.to_jshtml())
    """

    # ----- basic field constants (not user-tunable) -----
    _MAX_FIELD_X: float = 120.0
    _MAX_FIELD_Y: float = 53.3
    _MAX_FIELD_PLAYERS: int = 22  # pre-allocate enough artists

    def __init__(
        self,
        play_df: pd.DataFrame,
        precomputed_df: pd.DataFrame,
        grid,
        plot_size_len: float = 16.0,
        *,
        # contours
        show_contours: bool = True,
        contour_levels: int = 20,
        contour_alpha: float = 0.30,
        off_cmap: str = "Reds",
        on_cmap: str = "Blues",
        # arrows
        arrow_scale: float = 2.0,
        arrow_head_width: float = 0.50,
        arrow_head_length: float = 1.00,
        arrow_length_boost: float = 1.35,
        arrow_tail_width: float = 0.20,
        # orientation fallback (used ONLY if a row lacks stored bias)
        orientation_bias_deg: float = 0.0,
        # view
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        grid_alpha: float = 0.10,
        title: str = "Play Animation with Influence",
    ) -> None:

        # --- cache dataframes (avoid chained assignment issues) ---
        self._frame_data = play_df.copy()
        self._precomputed_df = precomputed_df.copy()

        # --- grid: accept either Grid-like (.X/.Y) or (X, Y) tuple ---
        try:
            self._X, self._Y = grid.X, grid.Y
        except AttributeError:
            self._X, self._Y = grid

        # --- frame order (FuncAnimation will iterate by index) ---
        self._frames = sorted(self._frame_data["frameId"].unique())

        # --- style / options (stored as floats for safety) ---
        self.show_contours = bool(show_contours)
        self.contour_levels = int(contour_levels)
        self.contour_alpha = float(contour_alpha)
        self.off_cmap = str(off_cmap)
        self.on_cmap = str(on_cmap)

        self.arrow_scale = float(arrow_scale)
        self.arrow_head_width = float(arrow_head_width)
        self.arrow_head_length = float(arrow_head_length)
        self.arrow_length_boost = float(arrow_length_boost)
        self.arrow_tail_width = float(arrow_tail_width)

        self.orientation_bias_deg = float(orientation_bias_deg)
        self.xlim = xlim
        self.ylim = ylim
        self.grid_alpha = float(grid_alpha)
        self._title = str(title)

        # --- animation plumbing ---
        self._stream = self._data_stream()

        # --- figure/axes (keep aspect ~ field) ---
        aspect = self._MAX_FIELD_Y / self._MAX_FIELD_X
        self._fig = plt.figure(figsize=(plot_size_len, plot_size_len * aspect))
        self._ax_field = plt.gca()

        # --- artist storage ---
        self._scat_jersey_list: list = []
        self._scat_number_list: list = []
        self._scat_name_list: list = []
        self._a_dir_list: list = []
        self._a_or_list: list = []
        self._contours: list = []

        # --- start the animation; we close figure to avoid duplicate displays in notebooks ---
        self.ani = animation.FuncAnimation(
            self._fig,
            self._update,
            frames=len(self._frames),
            init_func=self._setup_plot,
            blit=False,
        )
        plt.close()

    # Public accessors for convenience
    @property
    def fig(self):  # noqa: D401
        """Matplotlib Figure used by the animation."""
        return self._fig

    @property
    def ax(self):  # noqa: D401
        """Matplotlib Axes used by the animation."""
        return self._ax_field

    # ------------------------------ helpers ------------------------------ #

    @staticmethod
    def _theta_from_tracking(deg: float, bias_deg: float) -> float:
        """Map tracking degrees (0°=north, CW) → matplotlib radians (0 rad=+X, CCW)."""
        return np.deg2rad((90.0 - float(deg) + float(bias_deg)) % 360.0)

    @staticmethod
    def _hide_axes(ax: plt.Axes, max_x: float, max_y: float) -> None:
        """Hide axis ticks/labels and set field limits."""
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim([0.0, max_x])
        ax.set_ylim([0.0, max_y])

    def _data_stream(self) -> Iterable[pd.DataFrame]:
        """Yield the rows for each frame in order (simple generator)."""
        for frame in self._frames:
            yield self._frame_data[self._frame_data["frameId"] == frame]

    def _setup_plot(self):
        """Initial draw: field lines, empty collections, and pre-allocated texts/arrows."""
        # Field background
        self._hide_axes(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        for x in range(10, int(self._MAX_FIELD_X), 10):
            self._ax_field.axvline(x, color="k", linestyle="-", alpha=0.05)

        # One PathCollection per entity type (we update offsets every frame)
        self._scat_ball = self._ax_field.scatter([], [], s=100, color="black", zorder=6)
        self._scat_offense = self._ax_field.scatter([], [], s=500, color="blue", edgecolors="k", zorder=5)
        self._scat_defense = self._ax_field.scatter([], [], s=500, color="red", edgecolors="k", zorder=5)

        # Pre-allocate text + arrows, so we mutate them instead of creating new artists each frame
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_field.text(0, 0, "", ha="center", va="center", c="white"))
            self._scat_number_list.append(self._ax_field.text(0, 0, "", ha="center", va="center", c="black"))
            self._scat_name_list.append(self._ax_field.text(0, 0, "", ha="center", va="center", c="black"))
            self._a_dir_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color="k")))
            self._a_or_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color="grey", width=2)))

        # Title & view
        self._ax_field.set_title(self._title)
        if self.xlim:
            self._ax_field.set_xlim(*self.xlim)
        if self.ylim:
            self._ax_field.set_ylim(*self.ylim)
        self._ax_field.grid(alpha=self.grid_alpha)

        return (
            self._scat_ball,
            self._scat_offense,
            self._scat_defense,
            *self._scat_jersey_list,
        )

    @staticmethod
    def _clear_contours_safe(contours_list: list) -> None:
        """Clear contour artists robustly across Matplotlib versions."""
        for contour in contours_list:
            try:
                if hasattr(contour, "collections"):
                    for col in contour.collections:
                        col.remove()
                elif hasattr(contour, "remove"):
                    contour.remove()
                else:
                    for artist in contour:
                        if hasattr(artist, "remove"):
                            artist.remove()
            except (TypeError, AttributeError, ValueError):
                # If a collection has already been removed, keep going
                continue
        contours_list.clear()

    @staticmethod
    def _set_offsets_safely(coll, xy: np.ndarray, size_each: float) -> None:
        """
        Update a PathCollection's offsets and sizes so "ghost" points never linger.

        - When empty, set *both* offsets and sizes to empty arrays.
        - When non-empty, set offsets and match the collection size length to N.
        """
        if xy.size == 0:
            coll.set_offsets(np.empty((0, 2)))
            coll.set_sizes(np.array([]))
        else:
            coll.set_offsets(xy)
            coll.set_sizes(np.full(xy.shape[0], float(size_each)))

    # ------------------------------ animation step ------------------------------ #

    def update(self, _frame_index):
        try:
            pos_df = next(self._stream)
        except StopIteration:
            self._stream = self.data_stream()
            pos_df = next(self._stream)
    
        frame_id = int(pos_df.frameId.iloc[0])
    
        # split
        offense_df = pos_df[pos_df.is_off == 1]
        defense_df = pos_df[pos_df.is_off == 0]
        ball_df = pos_df[pos_df.team == 'football'] if 'team' in pos_df.columns \
                  else pos_df[pos_df.displayName.str.lower() == 'football']
    
        # update scatters safely (prevents ghost dots)
        to_xy = lambda d: np.column_stack((d["x"].to_numpy(float), d["y"].to_numpy(float))) if not d.empty else np.empty((0, 2))
        self._set_offsets_safely(self._scat_offense, to_xy(offense_df), 500.0)
        self._set_offsets_safely(self._scat_defense, to_xy(defense_df), 500.0)
        self._set_offsets_safely(self._scat_ball,    to_xy(ball_df),    100.0)
    
        self._ax_field.set_title(f"{self._title} — Frame {frame_id}")
    
        # clear old contours
        self.clear_contours_safe(self._contours)
    
        # plot densities for this frame
        if self.show_contours:
            frame_density_df = self._precomputed_df[self._precomputed_df.frameId == frame_id]
            for _, prow in frame_density_df.iterrows():
                Z = prow.get('density', None)
                if Z is None:
                    continue
                cmap = self.on_cmap if prow.get('is_off', 0) == 1 else self.off_cmap
                Z_masked = np.where(Z > 0.01, Z, np.nan)
                cont = self._ax_field.contourf(self._X, self._Y, Z_masked,
                                               cmap=cmap, levels=self.contour_levels, alpha=self.contour_alpha)
                self._contours.append(cont)
    
        # update per-player labels + arrows
        labeled = pos_df[pos_df.jerseyNumber.notnull()].reset_index(drop=True)
        for idx in range(min(len(labeled), self._MAX_FIELD_PLAYERS)):
            row = labeled.iloc[idx]
            x0, y0 = float(row["x"]), float(row["y"])
    
            # jersey/number/name
            self._scat_jersey_list[idx].set_position((x0, y0))
            self._scat_jersey_list[idx].set_text(row["position"] if "position" in row else "")
            self._scat_number_list[idx].set_position((x0, y0 + 1.5))
            try:
                self._scat_number_list[idx].set_text(int(row["jerseyNumber"]))
            except Exception:
                self._scat_number_list[idx].set_text("")
            self._scat_name_list[idx].set_position((x0, y0 - 1.5))
            name = row["displayName"].split()[-1] if isinstance(row["displayName"], str) else ""
            self._scat_name_list[idx].set_text(name)
    
            # Find matching precomputed row once
            match = self._precomputed_df[
                (self._precomputed_df.frameId == frame_id) & (self._precomputed_df.nflId == row["nflId"])
            ]
    
            # ALWAYS resolve row_bias first so it's available for both arrows
            row_bias = self.orientation_bias_deg
            if not match.empty and ("orientation_bias_deg" in match.columns):
                try:
                    row_bias = float(match.iloc[0].get("orientation_bias_deg", row_bias))
                except Exception:
                    pass
    
            # Resolve heading (direction) theta
            theta_rad = match.iloc[0].get("theta_rad", None) if not match.empty else None
            if theta_rad is None:
                # fall back to tracking dir (or 'direction') with the same bias mapping
                dir_deg = float(row["dir"]) if "dir" in row else float(row.get("direction", 0.0))
                theta_rad = self._theta_from_tracking(dir_deg, row_bias)
    
            # arrow length (modest speed scaling + boost)
            spd = float(row["s"]) if "s" in row else float(row.get("speed", 0.0))
            L = self.arrow_length_boost * self.arrow_scale * (1.0 + min(max(spd, 0.0), 11.3) / 11.3)
    
            # replace "direction" (black) arrow
            self._a_dir_list[idx].remove()
            self._a_dir_list[idx] = self._ax_field.add_patch(
                Arrow(x0, y0, L*np.cos(theta_rad), L*np.sin(theta_rad),
                      color='k', width=self.arrow_tail_width, lw=0, alpha=1.0, zorder=7)
            )
    
            # replace "orientation" (grey) arrow — optional
            self._a_or_list[idx].remove()
            if ("o" in row) and (row["o"] == row["o"]):  # pd.notna without importing pandas here
                orient_theta = self._theta_from_tracking(float(row["o"]), row_bias)
                self._a_or_list[idx] = self._ax_field.add_patch(
                    Arrow(x0, y0, (L*0.6)*np.cos(orient_theta), (L*0.6)*np.sin(orient_theta),
                          color='grey', width=self.arrow_tail_width*1.2, alpha=0.95, zorder=6)
                )
            else:
                self._a_or_list[idx] = self._ax_field.add_patch(Arrow(0, 0, 0, 0, color='grey', width=0))
    
        # Hide any unused artists
        for idx in range(len(labeled), self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list[idx].set_text('')
            self._scat_number_list[idx].set_text('')
            self._scat_name_list[idx].set_text('')
            self._a_dir_list[idx].remove()
            self._a_or_list[idx].remove()
    
        return (self._scat_ball, self._scat_offense, self._scat_defense,
                *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list,
                *self._a_dir_list, *self._a_or_list)
