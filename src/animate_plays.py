# src/animate_plays.py
from __future__ import annotations

from dataclasses import dataclass  # (you can remove if unused elsewhere)
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from matplotlib import animation

__all__ = ["AnimatePlayWithDensity"]


class AnimatePlayWithDensity:
    """
    Animate a play using **precomputed per-player density grids**.

    Tracking angle convention:
      - 0° = north (+Y), increasing clockwise to 360°
    Arrow mapping to Matplotlib:
      - theta = radians(90 - deg + bias)

    Expected inputs
    ---------------
    play_df : DataFrame (all frames for a play)
      Required columns: frameId, x, y, is_off (1/0), jerseyNumber, displayName
      Heading/speed: dir (deg), s (yd/s)
      Ball: team=='football' or displayName=='football'

    precomputed_df : DataFrame (one row per player per frame)
      Required: frameId, nflId, x, y, is_off, density (np.ndarray[ny, nx])
      Optional: theta_rad (float), orientation_bias_deg (float)

    grid :
      Either a (X, Y) tuple or an object with attributes .X and .Y
    """

    # basic field constants
    _MAX_FIELD_X: float = 120.0
    _MAX_FIELD_Y: float = 53.3
    _MAX_FIELD_PLAYERS: int = 22

    def __init__(
        self,
        play_df: pd.DataFrame,
        precomputed_df: pd.DataFrame,
        grid,                              # (X, Y) tuple OR a Grid-like with .X/.Y
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
        # orientation fallback (only if row lacks stored bias)
        orientation_bias_deg: float = 0.0,
        # view
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        grid_alpha: float = 0.10,
        title: str = "Play Animation with Influence",
    ) -> None:

        # cache dataframes
        self._frame_data = play_df.copy()
        self._precomputed_df = precomputed_df.copy()

        # grid: accept either Grid-like (.X/.Y) or (X, Y) tuple
        try:
            self._X, self._Y = grid.X, grid.Y
        except AttributeError:
            self._X, self._Y = grid

        # frame order for the animation
        self._frames = sorted(self._frame_data["frameId"].unique())

        # style / options
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

        # animation plumbing
        self._stream = self._data_stream()

        # figure/axes
        aspect = self._MAX_FIELD_Y / self._MAX_FIELD_X
        self._fig = plt.figure(figsize=(plot_size_len, plot_size_len * aspect))
        self._ax_field = plt.gca()

        # artist storage
        self._scat_jersey_list: list = []
        self._scat_number_list: list = []
        self._scat_name_list: list = []
        self._a_dir_list: list = []
        self._a_or_list: list = []
        self._contours: list = []

        # start the animation (use the private _update/_setup_plot methods)
        self.ani = animation.FuncAnimation(
            self._fig,
            self._update,
            frames=len(self._frames),
            init_func=self._setup_plot,
            blit=False,
        )
        plt.close()

    # ----------------------- public accessors -----------------------
    @property
    def fig(self):
        """Matplotlib Figure used by the animation."""
        return self._fig

    @property
    def ax(self):
        """Matplotlib Axes used by the animation."""
        return self._ax_field

    # ----------------------- static helpers ------------------------
    @staticmethod
    def _theta_from_tracking(deg: float, bias_deg: float) -> float:
        """Map tracking degrees (0°=north, CW) → matplotlib radians (0 rad=+X, CCW)."""
        return np.deg2rad((90.0 - float(deg) + float(bias_deg)) % 360.0)

    @staticmethod
    def _hide_axes(ax: plt.Axes, max_x: float, max_y: float) -> None:
        """Hide ticks/labels and lock field bounds."""
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim([0.0, max_x])
        ax.set_ylim([0.0, max_y])

    # ----------------------- data stream ---------------------------
    def _data_stream(self) -> Iterable[pd.DataFrame]:
        """Yield the rows for each frame in order (simple generator)."""
        for frame in self._frames:
            yield self._frame_data[self._frame_data["frameId"] == frame]

    # ----------------------- init draw ------------------------------
    def _setup_plot(self):
        """Initial draw: field lines, empty collections, and pre-allocated texts/arrows."""
        self._hide_axes(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        for x in range(10, int(self._MAX_FIELD_X), 10):
            self._ax_field.axvline(x, color="k", linestyle="-", alpha=0.05)

        # One PathCollection per entity type (we update offsets every frame)
        self._scat_ball = self._ax_field.scatter([], [], s=100, color="black", zorder=6)
        self._scat_offense = self._ax_field.scatter([], [], s=500, color="blue", edgecolors="k", zorder=5)
        self._scat_defense = self._ax_field.scatter([], [], s=500, color="red", edgecolors="k", zorder=5)

        # Pre-allocate text + arrows (mutated in-place per frame)
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

        return (self._scat_ball, self._scat_offense, self._scat_defense, *self._scat_jersey_list)

    # ----------------------- contour cleanup -----------------------
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
                continue
        contours_list.clear()

    # ----------------------- PathCollection helper -----------------
    @staticmethod
    def _set_offsets_safely(coll, xy: np.ndarray, size_each: float) -> None:
        """
        Update a PathCollection's offsets and sizes so "ghost" points never linger.
        When empty: set *both* offsets and sizes to empty arrays.
        """
        if xy.size == 0:
            coll.set_offsets(np.empty((0, 2)))
            coll.set_sizes(np.array([]))
        else:
            coll.set_offsets(xy)
            coll.set_sizes(np.full(xy.shape[0], float(size_each)))

    # ----------------------- animation step ------------------------
    def _update(self, _frame_index):
        # get next frame; reset generator cleanly on exhaustion
        try:
            pos_df = next(self._stream)
        except StopIteration:
            self._stream = self._data_stream()
            pos_df = next(self._stream)

        frame_id = int(pos_df.frameId.iloc[0])

        # split entities
        offense_df = pos_df[pos_df.is_off == 1]
        defense_df = pos_df[pos_df.is_off == 0]
        ball_df = (
            pos_df[pos_df.team == "football"] if "team" in pos_df.columns
            else pos_df[pos_df.displayName.str.lower() == "football"]
        )

        # update scatters safely (no ghosts)
        to_xy = (
            lambda d: np.column_stack((d["x"].to_numpy(float), d["y"].to_numpy(float)))
            if not d.empty else np.empty((0, 2))
        )
        self._set_offsets_safely(self._scat_offense, to_xy(offense_df), 500.0)
        self._set_offsets_safely(self._scat_defense, to_xy(defense_df), 500.0)
        self._set_offsets_safely(self._scat_ball, to_xy(ball_df), 100.0)

        self._ax_field.set_title(f"{self._title} — Frame {frame_id}")

        # clear and (optionally) redraw contours
        self._clear_contours_safe(self._contours)
        if self.show_contours:
            frame_density_df = self._precomputed_df[self._precomputed_df.frameId == frame_id]
            for _, prow in frame_density_df.iterrows():
                Z = prow.get("density", None)
                if Z is None:
                    continue
                cmap = self.on_cmap if prow.get("is_off", 0) == 1 else self.off_cmap
                Z_masked = np.where(Z > 0.01, Z, np.nan)
                cont = self._ax_field.contourf(
                    self._X, self._Y, Z_masked,
                    cmap=cmap, levels=self.contour_levels, alpha=self.contour_alpha,
                )
                self._contours.append(cont)

        # labels + arrows
        labeled = pos_df[pos_df.jerseyNumber.notnull()].reset_index(drop=True)
        for idx in range(min(len(labeled), self._MAX_FIELD_PLAYERS)):
            row = labeled.iloc[idx]
            x0, y0 = float(row["x"]), float(row["y"])

            # text
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

            # find any matching precomputed row for bias / theta
            match = self._precomputed_df[
                (self._precomputed_df.frameId == frame_id) & (self._precomputed_df.nflId == row["nflId"])
            ]

            # resolve row_bias once (used for both arrows)
            row_bias = self.orientation_bias_deg
            if not match.empty and ("orientation_bias_deg" in match.columns):
                try:
                    row_bias = float(match.iloc[0].get("orientation_bias_deg", row_bias))
                except Exception:
                    pass

            # direction angle
            theta_rad = match.iloc[0].get("theta_rad", None) if not match.empty else None
            if theta_rad is None:
                dir_deg = float(row["dir"]) if "dir" in row else float(row.get("direction", 0.0))
                theta_rad = self._theta_from_tracking(dir_deg, row_bias)

            # length (speed scaled + boost)
            spd = float(row["s"]) if "s" in row else float(row.get("speed", 0.0))
            L = self.arrow_length_boost * self.arrow_scale * (1.0 + min(max(spd, 0.0), 11.3) / 11.3)

            # draw direction arrow (black)
            self._a_dir_list[idx].remove()
            self._a_dir_list[idx] = self._ax_field.add_patch(
                Arrow(
                    x0, y0, L * np.cos(theta_rad), L * np.sin(theta_rad),
                    color="k", width=self.arrow_tail_width, lw=0, alpha=1.0, zorder=7,
                )
            )

            # draw orientation arrow (grey) if available
            self._a_or_list[idx].remove()
            if ("o" in row) and (row["o"] == row["o"]):  # pd.notna without importing pandas here
                orient_theta = self._theta_from_tracking(float(row["o"]), row_bias)
                self._a_or_list[idx] = self._ax_field.add_patch(
                    Arrow(
                        x0, y0, (L * 0.6) * np.cos(orient_theta), (L * 0.6) * np.sin(orient_theta),
                        color="grey", width=self.arrow_tail_width * 1.2, alpha=0.95, zorder=6,
                    )
                )
            else:
                self._a_or_list[idx] = self._ax_field.add_patch(Arrow(0, 0, 0, 0, color="grey", width=0))

        # hide any unused artists
        for idx in range(len(labeled), self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list[idx].set_text("")
            self._scat_number_list[idx].set_text("")
            self._scat_name_list[idx].set_text("")
            self._a_dir_list[idx].remove()
            self._a_or_list[idx].remove()

        return (
            self._scat_ball, self._scat_offense, self._scat_defense,
            *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list,
            *self._a_dir_list, *self._a_or_list,
        )
