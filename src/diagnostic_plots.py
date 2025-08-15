from typing import Iterable, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_dir_orientation_small_multiples(
    df: pd.DataFrame,
    *,
    frame_ids: Optional[Sequence[int]] = None,
    base_frame: Optional[int] = None,
    offsets: Sequence[int] = (0, 29, 59, 89, 119, 149, 179),
    # columns
    frame_col: str = "frameId",
    x_col: str = "x",
    y_col: str = "y",
    dir_col: str = "dir",   # 0° = north/up (+Y), degrees increase clockwise
    ori_col: str = "o",     # same convention as dir (if present)
    speed_col: str = "s",   # yards/s (optional; used only for arrow scaling)
    # field + styling
    field_x_max: float = 120.0,
    field_y_max: float = 53.3,
    cols: int = 4,
    arrow_len: float = 6.0,         # base arrow length in yards
    scale_with_speed: bool = True,  # modestly scale arrow by speed
    dir_color: str = "tab:blue",
    ori_color: str = "tab:orange",
    grid_alpha: float = 0.15,
    title_prefix: str = "Dir vs Orientation",
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot direction vs. orientation arrows for a *single player's* time series as
    a grid of small-multiple panels.

    Each panel shows:
      - the player's (x, y) position at a given frame
      - a direction arrow (``dir_col``) in ``dir_color``
      - an orientation arrow (``ori_col``) in ``ori_color`` if available

    Angle convention
    ----------------
    This function assumes the tracking convention:
      * 0° = due north (up, +Y)
      * angles increase clockwise up to 360°
    To draw with Matplotlib (where 0 rad = +X and angles increase CCW), we convert:
      ``theta = radians(90 - deg)``

    Parameters
    ----------
    df
        DataFrame containing at least ``frame_col, x_col, y_col, dir_col``.
        It should contain rows for **one** player (pre-filter upstream).
    frame_ids
        Exact frame IDs to plot. If provided, takes precedence over ``base_frame``/``offsets``.
    base_frame
        If ``frame_ids`` is None, we build frames as ``base_frame + offsets``.
        If also None, we default to ``df[frame_col].min()``.
    offsets
        Relative frame offsets from ``base_frame`` when ``frame_ids`` is None.
    frame_col, x_col, y_col, dir_col, ori_col, speed_col
        Column names in ``df``.
    field_x_max, field_y_max
        Field dimensions (yards).
    cols
        Number of columns in the subplot grid.
    arrow_len
        Base arrow length in yards (before any speed scaling).
    scale_with_speed
        If True, arrow length is multiplied by a small factor based on speed:
        ``0.75 + min(max(speed, 0), 11.3) / 22.6``.
    dir_color, ori_color
        Arrow colors for direction and orientation.
    grid_alpha
        Alpha for the background grid lines.
    title_prefix
        Suptitle prefix for the figure.
    show
        If True, call ``plt.show()`` before returning.

    Returns
    -------
    (fig, axes)
        The Matplotlib figure and a **flattened** NumPy array of Axes.

    Raises
    ------
    ValueError
        If none of the requested frames are present in ``df``.
    KeyError
        If required columns are missing from ``df``.

    Notes
    -----
    * This diagnostic is intended for a **single player's** rows. If your ``df``
      contains multiple players at the same frame, the first row for that frame
      will be used. Prefer pre-filtering: ``df[df['nflId'] == some_player_id]``.
    * Orientation is optional—if ``ori_col`` is missing or NaN, only direction
      arrows are drawn.
    """
    # ---- basic column validation (lightweight, helpful errors) -------------
    required = {frame_col, x_col, y_col, dir_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # ---- choose frames to plot ---------------------------------------------
    if frame_ids is None:
        if base_frame is None:
            # default to the earliest frame present
            base_frame = int(df[frame_col].min())
        frame_ids = [base_frame + off for off in offsets]

    frames_available = set(df[frame_col].unique())
    frames_to_plot = [f for f in frame_ids if f in frames_available]
    missing_frames = [f for f in frame_ids if f not in frames_available]
    if not frames_to_plot:
        raise ValueError("None of the requested frames are present in the DataFrame.")
    if missing_frames:
        # Non-fatal: just let the user know via stdout
        print(f"Note: skipping missing frames: {missing_frames}")

    # ---- angle converter: tracking → Matplotlib -----------------------------
    def to_theta(deg: float) -> float:
        # tracking convention (0° up, clockwise) → matplotlib (0 rad right, CCW)
        return np.deg2rad(90.0 - deg)

    # ---- figure layout ------------------------------------------------------
    n = len(frames_to_plot)
    rows = int(np.ceil(n / cols))
    aspect = field_y_max / field_x_max
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows * aspect))
    axes = np.array(axes).reshape(-1)

    # ---- render each panel --------------------------------------------------
    for ax, fid in zip(axes, frames_to_plot):
        # Expect exactly one row per frame for this player; if multiple, take first
        row = df.loc[df[frame_col] == fid]
        if row.empty:
            ax.axis("off")
            continue
        r = row.iloc[0]

        # Extract scalars safely
        x0, y0 = float(r[x_col]), float(r[y_col])
        dir_deg = float(r[dir_col]) if pd.notna(r[dir_col]) else None
        ori_deg = float(r[ori_col]) if (ori_col in r and pd.notna(r[ori_col])) else None
        spd = float(r[speed_col]) if (speed_col in r and pd.notna(r[speed_col])) else 0.0

        # modest speed scaling (keeps arrows readable)
        scale = (0.75 + min(max(spd, 0.0), 11.3) / 22.6) if scale_with_speed else 1.0
        L = arrow_len * scale

        # field window + yard lines
        ax.set_xlim(0, field_x_max)
        ax.set_ylim(0, field_y_max)
        for x in range(10, int(field_x_max), 10):
            ax.axvline(x, color="k", lw=1, alpha=0.06)
        ax.scatter(x0, y0, s=60, c="k", zorder=5)

        # direction arrow (blue)
        if dir_deg is not None:
            th = to_theta(dir_deg)
            ax.arrow(
                x0, y0,
                L * np.cos(th), L * np.sin(th),
                head_width=1.2, head_length=2.0,
                fc=dir_color, ec=dir_color, lw=1.5, zorder=6
            )

        # orientation arrow (orange, if present)
        if ori_deg is not None:
            th_o = to_theta(ori_deg)
            ax.arrow(
                x0, y0,
                L * np.cos(th_o), L * np.sin(th_o),
                head_width=1.2, head_length=2.0,
                fc=ori_color, ec=ori_color, lw=1.5, alpha=0.8, zorder=6
            )

        # panel title with exact numeric values
        t_dir = f"{dir_deg:.2f}°" if dir_deg is not None else "NA"
        t_ori = f"{ori_deg:.2f}°" if ori_deg is not None else "NA"
        ax.set_title(f"Frame {fid}  |  dir={t_dir}  |  o={t_ori}", fontsize=10)

        # cosmetic
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=grid_alpha)

    # Turn off any extra (unused) axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix}: direction (blue) vs orientation (orange) for a single player",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if show:
        plt.show()

    return fig, axes
