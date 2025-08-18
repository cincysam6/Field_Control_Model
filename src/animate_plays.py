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

def animate_pitch_control_with_players(
    *,
    player_df: pd.DataFrame,
    pc_by_frame: dict,                  # {frameId: pitch_control_df}
    frames: list,                       # ordered list of frameIds to animate
    heading_col: str = "direction",     # compute_* functions store heading here
    arrow_scale: float = 5.0,
    constant_arrow: bool = False,
    xlim=(60, 110),
    ylim=(5, 45),
    cmap="coolwarm",
    levels: int = 20,
    surface_alpha: float = 0.8,
    figsize=(16, 8),
    show_colorbar: bool = True,
):
    """Animate team pitch control + player dots/numbers/names/arrows."""

    def theta_from_tracking(deg: float) -> float:
        # 0° = north/up (+Y), clockwise → matplotlib (0 rad = +X, CCW)
        return np.deg2rad((90.0 - float(deg)) % 360.0)

    # ---------- figure & static scaffolding ----------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("X (yards)"); ax.set_ylabel("Y (yards)")
    ax.grid(alpha=0.4)
    for xv in range(10, 120, 10):
        ax.axvline(xv, color="k", lw=1, alpha=0.06)

    f0 = frames[0]
    Z0 = pc_by_frame[f0].values
    X = pc_by_frame[f0].columns.values
    Y = pc_by_frame[f0].index.values

    # draw first surface and capture the artists explicitly
    cs0 = ax.contourf(X, Y, Z0, cmap=cmap, levels=levels, alpha=surface_alpha)
    contour_artists = list(getattr(cs0, "collections", [])) or (cs0 if hasattr(cs0, "__iter__") else [])
    cbar = None
    if show_colorbar:
        cbar = fig.colorbar(cs0, ax=ax, label="Field Control (defense → 1.0)")

    title = ax.set_title(f"Team Field Control — Frame {f0}")

    # holders we’ll refresh each frame
    scatters, texts_num, texts_name, arrows = [], [], [], []

    def _clear_artists(artists):
        # Safely remove any list of artists
        for a in list(artists):
            try:
                a.remove()
            except Exception:
                pass
        artists.clear()

    def _draw_players(frame_id: int):
        _clear_artists(scatters)
        _clear_artists(texts_num)
        _clear_artists(texts_name)
        _clear_artists(arrows)

        f = player_df[player_df["frameId"] == frame_id]
        for row in f.itertuples(index=False):
            x0 = float(getattr(row, "x"))
            y0 = float(getattr(row, "y"))
            is_off_val = getattr(row, "is_off", 0)
            try:
                is_off_bool = bool(int(is_off_val))
            except Exception:
                is_off_bool = False
            color = "blue" if is_off_bool else "red"

            sc = ax.scatter(x0, y0, color=color, edgecolor="black", s=120, zorder=5)
            scatters.append(sc)

            jn = getattr(row, "jerseyNumber", None)
            if jn is not None and not (isinstance(jn, float) and np.isnan(jn)):
                tnum = ax.text(x0, y0 + 1.0, f"{int(jn)}", fontsize=9, ha="center", color="black", zorder=6)
                texts_num.append(tnum)

            name = getattr(row, "displayName", "")
            if isinstance(name, str) and name:
                tname = ax.text(x0, y0 - 1.4, name, fontsize=8, ha="center", color="black", zorder=6)
                texts_name.append(tname)

            # heading
            deg = getattr(row, heading_col, getattr(row, "dir", None))
            if deg is not None and not (isinstance(deg, float) and np.isnan(deg)):
                th = theta_from_tracking(float(deg))
                spd = getattr(row, "speed", getattr(row, "s", 0.0))
                try:
                    spd = float(spd)
                except Exception:
                    spd = 0.0
                L = (arrow_scale if constant_arrow else arrow_scale * (spd / 11.3))
                ar = ax.arrow(
                    x0, y0, L*np.cos(th), L*np.sin(th),
                    head_width=0.375, head_length=0.75, fc="black", ec="black", zorder=7
                )
                arrows.append(ar)

    _draw_players(f0)

    def _update(i):
        fid = frames[i]
        title.set_text(f"Team Field Control — Frame {fid}")

        # remove previous contour artists robustly
        _clear_artists(contour_artists)

        # draw new surface and refresh the stored artist list
        cs = ax.contourf(X, Y, pc_by_frame[fid].values, cmap=cmap, levels=levels, alpha=surface_alpha)
        contour_artists[:] = list(getattr(cs, "collections", [])) or (cs if hasattr(cs, "__iter__") else [])

        if cbar is not None:
            try:
                cbar.update_normal(cs)
            except Exception:
                # fallback: replace colorbar mappable
                cbar.mappable = cs
                cbar.draw_all()

        _draw_players(fid)
        # return all artists for blit=False (harmless) or completeness
        return contour_artists + scatters + texts_num + texts_name + arrows + [title]

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=120, blit=False)
    plt.close(fig)
    return ani


def animate_pitch_control_with_players_fast(
    *,
    player_df: pd.DataFrame,
    pc_by_frame: Dict[int, pd.DataFrame],   # {frameId: pitch_control_df (index=Y, columns=X)}
    frames: List[int],                      # ordered list of frameIds to animate
    heading_col: str = "direction",         # fallback to "dir" if missing

    # ---- constant arrows (no speed scaling) ----
    arrow_len: float = 2.0,                 # yards; short stem off the dot
    # small head, smaller than s=120 scatter
    head_length_pt: float = 4.0,            # points
    head_width_pt: float  = 3.0,            # points
    shaft_width_ax: float = 0.006,          # axes fraction; thin

    # view & style
    xlim: Tuple[float, float] = (60, 110),
    ylim: Tuple[float, float] = (5, 45),
    cmap: str = "coolwarm",
    surface_alpha: float = 0.85,
    figsize: Tuple[int, int] = (16, 8),
    show_colorbar: bool = True,
    show_labels: bool = True,               # jersey numbers + names
    max_players: int = 22,                  # text labels capacity
    max_arrows: int = 30,                   # quiver capacity (>= players shown per frame)
    interval_ms: int = 80,                  # ~12.5 fps
) -> animation.FuncAnimation:
    """
    Fast animation of team pitch control + players using imshow (surface) and a fixed-size
    quiver for arrows. Arrows are constant length with a small head (no speed scaling).
    """

    def theta_from_tracking(deg: float) -> float:
        # 0° = +Y (north), clockwise → Matplotlib radians (0 rad = +X, CCW)
        return np.deg2rad((90.0 - float(deg)) % 360.0)

    # ---------- figure & static scaffolding ----------
    f0 = frames[0]
    z0 = pc_by_frame[f0].values
    X = pc_by_frame[f0].columns.to_numpy(dtype=float)
    Y = pc_by_frame[f0].index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("X (yards)"); ax.set_ylabel("Y (yards)")
    ax.grid(alpha=0.4)
    for xv in range(10, 120, 10):
        ax.axvline(xv, color="k", lw=1, alpha=0.06)

    im = ax.imshow(
        z0,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap=cmap,
        alpha=surface_alpha,
        interpolation="nearest",
        aspect="auto",
        vmin=0.0, vmax=1.0,
        zorder=0,
    )
    cbar = fig.colorbar(im, ax=ax, label="Field Control (defense → 1.0)") if show_colorbar else None
    title = ax.set_title(f"Team Field Control — Frame {f0}")

    scat_off = ax.scatter([], [], s=120, c="blue", edgecolors="black", zorder=10)
    scat_def = ax.scatter([], [], s=120, c="red",  edgecolors="black", zorder=10)
    scat_ball = ax.scatter([], [], s=90,  c="black", zorder=12)

    # ---------- FIXED-SIZE QUIVER (off-screen padding, no NaNs) ----------
    OFF = 1e9  # off-canvas marker
    q_offsets = np.full((max_arrows, 2), OFF, dtype=float)
    q_U = np.zeros((max_arrows,), dtype=float)
    q_V = np.zeros((max_arrows,), dtype=float)
    quiv = ax.quiver(
        q_offsets[:, 0], q_offsets[:, 1], q_U, q_V,
        angles="xy", scale_units="xy", scale=1.0, pivot="tail",
        width=shaft_width_ax, color="black", linewidths=0.3,
        headlength=head_length_pt, headaxislength=head_length_pt*0.9, headwidth=head_width_pt,
        zorder=20
    )

    # Preallocate jersey numbers + names (Text)
    nums = [ax.text(0, 0, "", fontsize=9, ha="center", color="black", zorder=18, visible=False)
            for _ in range(max_players)]
    names = [ax.text(0, 0, "", fontsize=8, ha="center", color="black", zorder=18, visible=False)
             for _ in range(max_players)]

    # ---------- helpers ----------
    def _frame_slice(fid: int) -> pd.DataFrame:
        f = player_df[player_df["frameId"] == fid]
        if f.empty:
            return f
        if "displayName" in f.columns:
            f = f[f["displayName"].str.lower() != "football"]
        return f

    def _to_offsets(d: pd.DataFrame) -> np.ndarray:
        if d.empty:
            return np.empty((0, 2), dtype=float)
        return np.column_stack((d["x"].to_numpy(float), d["y"].to_numpy(float)))

    def _compute_vectors(d: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-length arrows in (U,V), independent of speed."""
        if d.empty:
            return np.empty((0,)), np.empty((0,))
        if heading_col not in d.columns and "dir" in d.columns:
            hdg = d["dir"].to_numpy(float)
        else:
            hdg = d[heading_col].to_numpy(float)
        th = np.deg2rad((90.0 - hdg) % 360.0)
        U = arrow_len * np.cos(th)
        V = arrow_len * np.sin(th)
        return U, V

    def _update_texts(d: pd.DataFrame):
        if not show_labels:
            for t in nums + names: t.set_visible(False)
            return
        n = min(len(d), max_players)
        if n == 0:
            for t in nums + names: t.set_visible(False)
            return
        sub = d.iloc[:n]
        xs = sub["x"].to_numpy(float); ys = sub["y"].to_numpy(float)

        jvals = sub.get("jerseyNumber", pd.Series([None]*n, index=sub.index)).to_numpy(object)
        for i in range(n):
            nums[i].set_position((xs[i], ys[i] + 1.0))
            try:
                txt = "" if jvals[i] is None or (isinstance(jvals[i], float) and np.isnan(jvals[i])) else str(int(jvals[i]))
            except Exception:
                txt = ""
            nums[i].set_text(txt); nums[i].set_visible(bool(txt))

        names_raw = sub.get("displayName", pd.Series([""]*n, index=sub.index)).astype(str).to_numpy()
        for i in range(n):
            nm = names_raw[i].split()[-1] if names_raw[i] else ""
            names[i].set_position((xs[i], ys[i] - 1.4))
            names[i].set_text(nm); names[i].set_visible(bool(nm))

        for i in range(n, max_players):
            nums[i].set_visible(False); names[i].set_visible(False)

    # ---------- animation callbacks ----------
    def _init():
        artists = [im, scat_off, scat_def, scat_ball, quiv, title]
        artists += nums + names
        return artists

    def _update(i: int):
        fid = frames[i]
        title.set_text(f"Team Field Control — Frame {fid}")
        im.set_data(pc_by_frame[fid].values)

        f = _frame_slice(fid)
        off = f[f.get("is_off", 0).astype(bool)]
        de  = f[~f.get("is_off", 0).astype(bool)]

        # scatters
        scat_off.set_offsets(_to_offsets(off))
        scat_def.set_offsets(_to_offsets(de))
        ball = player_df[(player_df["frameId"] == fid) &
                         (player_df.get("displayName", "").str.lower() == "football")]
        scat_ball.set_offsets(_to_offsets(ball))

        # -------- fixed-size quiver update (off-screen padding) --------
        all_df = pd.concat([off, de], axis=0) if (not off.empty or not de.empty) else pd.DataFrame(columns=["x","y"])
        n = min(len(all_df), max_arrows)
        if n > 0:
            offs = _to_offsets(all_df.iloc[:n])
            U,V = _compute_vectors(all_df.iloc[:n])
            # move all arrow slots off-screen, then fill first n
            q_offsets[:, :] = OFF
            q_U[:] = 0.0; q_V[:] = 0.0
            q_offsets[:n, :] = offs
            q_U[:n], q_V[:n] = U, V
        else:
            # no players: keep everything off-screen
            q_offsets[:, :] = OFF
            q_U[:] = 0.0; q_V[:] = 0.0

        quiv.set_offsets(q_offsets)
        quiv.set_UVC(q_U, q_V)

        _update_texts(f)

        artists = [im, scat_off, scat_def, scat_ball, quiv, title]
        artists += [t for t in nums + names if t.get_visible()]
        return artists

    ani = animation.FuncAnimation(
        fig, _update, init_func=_init, frames=len(frames),
        interval=interval_ms, blit=True
    )
    plt.close(fig)
    return ani
