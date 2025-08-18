from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from scipy.ndimage import gaussian_filter

import numpy as np
import pandas as pd
from src.field_control_model import PlayerInfluenceModel
# If you keep a canonical default in your package, you can import it like:
# from .defaults import model_kwargs as DEFAULT_MODEL_KWARGS
# For notebooks, you can pass `model_kwargs=` explicitly.


def compute_player_densities_dataframe(
    df: pd.DataFrame,
    min_frame: Optional[int] = None,
    max_frame: Optional[int] = None,
    *,
    frames: Optional[Sequence[int]] = None,
    player_ids: Optional[Union[int, Sequence[int]]] = None,
    # Model configuration (merged with your package-level `model_kwargs`)
    model_kwargs: Optional[Dict] = None,
    alpha_gamma: Optional[float] = None,          # if None → uses model.alpha_gamma
    exclude_football: bool = True,
    # Column names (override here if your schema differs)
    id_col: str = "nflId",
    name_col: str = "displayName",
    jersey_col: str = "jerseyNumber",
    x_col: str = "x",
    y_col: str = "y",
    dir_col: str = "dir",                         # 0° = north, clockwise
    speed_col: str = "s",                         # yards/s
    is_off_col: str = "is_off",
    dist_from_ball_col: str = "dist_from_football",
    # What to store back
    store_theta_rad: bool = True,                 # save the radians actually used for arrows
    store_bias_deg: bool = True,                  # save model.orientation_bias_deg used
    density_out_col: str = "density",             # name of the 2D array column
    # Convenience / safety
    drop_na_rows: bool = True,                    # drop rows with required NaNs
    quiet: bool = False,                          # suppress per-row error prints
) -> Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-player *individual* influence grids (Gaussian + Gamma mix) for a
    range of frames using `PlayerInfluenceModel`, returning one row per
    (frameId, playerId) with the full 2D density grid.

    This is the “single-player density” producer you can animate or inspect.

    Parameters
    ----------
    df :
        Tracking-like DataFrame containing at least:
        [frameId, {id_col}, {x_col}, {y_col}, {dir_col}, {speed_col}].
        If available, {dist_from_ball_col} improves the Gaussian shaping.
    min_frame, max_frame :
        Frame bounds (inclusive). Ignored if `frames` is provided.
    frames :
        Explicit list/sequence of frames to compute. Takes precedence over
        min/max.
    player_ids :
        Single id or list of ids to include. If None, all players are used.
    model_kwargs :
        Dict of kwargs for `PlayerInfluenceModel(...)`. If you keep a canonical
        `model_kwargs` in your package, pass it here to stay consistent.
    alpha_gamma :
        Override the model’s `alpha_gamma` for the Gamma component. If None,
        uses whatever the model was constructed with.
    exclude_football :
        If True, removes rows where `displayName` (or `name_col`) equals
        'football' (case-insensitive).
    id_col, name_col, jersey_col, x_col, y_col, dir_col, speed_col,
    is_off_col, dist_from_ball_col :
        Column names in `df`.
    store_theta_rad :
        If True, store the Matplotlib-ready heading (radians) used by the model
        for arrow plotting (`theta = radians(90 - deg)` with no extra bias).
    store_bias_deg :
        If True, store the model’s `orientation_bias_deg` used for this run.
        (We keep this off by default in your modeling; storing helps plotting
        functions reproduce arrow conventions if bias is ever non-zero.)
    density_out_col :
        Name of the column that will hold the 2D numpy array.
    drop_na_rows :
        If True, rows with NaN in required fields are skipped.
    quiet :
        If True, suppress per-row error messages and continue.

    Returns
    -------
    all_player_df : pd.DataFrame
        One row per (frameId, playerId) with:
        [frameId, id_col, name_col, jersey_col, x, y, speed, direction,
         is_off, dist_from_football, {density_out_col}, optional theta/bias].
    (X, Y) : Tuple[np.ndarray, np.ndarray]
        The grid used by the model (ny×nx arrays).

    Notes
    -----
    • Angles use the same mapping as your class:
        theta = radians(90 - deg)
      We *do not* apply additional bias unless your `model_kwargs` sets it.
    • For stability, we allow missing `dist_from_football` and default it to 0.0.
    • If you have many frames/players, consider chunking externally or adding a
      light progress print around the frame loop.
    """
    # --------- 0) Import the model (assumes it’s in your package namespace) -- #
    try:
        PlayerInfluenceModel  # type: ignore[name-defined]
    except NameError as e:
        raise NameError(
            "PlayerInfluenceModel must be defined/imported before calling compute_player_densities_dataframe()."
        ) from e

    # --------- 1) Frame selection ------------------------------------------- #
    if frames is not None:
        frames_to_do = sorted(set(int(f) for f in frames))
    else:
        # Fall back to min/max window
        if min_frame is None:
            min_frame = int(df["frameId"].min())
        if max_frame is None:
            max_frame = int(df["frameId"].max())
        if min_frame > max_frame:
            raise ValueError("min_frame must be <= max_frame.")
        present = df["frameId"].unique()
        frames_to_do = sorted(int(f) for f in present if min_frame <= f <= max_frame)

    if not frames_to_do:
        raise ValueError("No frames selected. Check your min/max or frames list.")

    # --------- 2) Build the model once ------------------------------------- #
    model = PlayerInfluenceModel(**(model_kwargs or {}))
    X, Y = model.grid.X, model.grid.Y
    use_alpha = model.alpha_gamma if alpha_gamma is None else float(alpha_gamma)

    # --------- 3) Slice dataframe to just what we need ---------------------- #
    fdf = df[df["frameId"].isin(frames_to_do)].copy()

    # Optional: restrict to players
    if player_ids is not None:
        if isinstance(player_ids, (int, np.integer, str)):
            player_ids = [player_ids]  # normalize
        fdf = fdf[fdf[id_col].isin(player_ids)].copy()

    # Optional: drop the ball row(s)
    if exclude_football and (name_col in fdf.columns):
        fdf = fdf[fdf[name_col].str.lower() != "football"].copy()

    # --------- 4) Validate columns / drop NA as requested ------------------- #
    required_cols = {id_col, x_col, y_col, dir_col, speed_col}
    missing = required_cols - set(fdf.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if drop_na_rows:
        fdf = fdf.dropna(subset=[x_col, y_col, dir_col, speed_col])

    # --------- 5) Compute densities row-by-row ------------------------------ #
    out_rows: List[Dict] = []
    for frame_id in frames_to_do:
        frame_rows = fdf[fdf["frameId"] == frame_id]
        if frame_rows.empty:
            continue

        for _, r in frame_rows.iterrows():
            try:
                pos = (float(r[x_col]), float(r[y_col]))
                dir_deg = float(r[dir_col])
                spd = float(r[speed_col])
                dball = float(r.get(dist_from_ball_col, 0.0)) if dist_from_ball_col in r else 0.0

                # Gamma anchor behind the player (keeps mode near the player)
                pos_off = model.compute_offset(pos, dir_deg, spd)

                # Full mixed density (Gaussian + Gamma), normalized internally
                Z = model.base_distribution(
                    pos_xy=pos,
                    pos_off_xy=pos_off,
                    direction_deg=dir_deg,
                    speed=spd,
                    dist_from_ball=dball,
                    alpha_gamma=use_alpha,
                )

                rec = {
                    "frameId": int(frame_id),
                    id_col: r[id_col],
                    name_col: r.get(name_col, None),
                    jersey_col: r.get(jersey_col, None),
                    "x": pos[0],
                    "y": pos[1],
                    "speed": spd,
                    "direction": dir_deg,                    # raw tracking dir (0°=north, CW)
                    is_off_col: r.get(is_off_col, None),
                    dist_from_ball_col: dball,
                    density_out_col: Z,
                }
                if store_theta_rad:
                    # store the exact radians used for arrows (no extra bias)
                    rec["theta_rad"] = model.theta_from_tracking(dir_deg, apply_bias=False)
                if store_bias_deg:
                    rec["orientation_bias_deg"] = model.orientation_bias_deg

                out_rows.append(rec)

            except Exception as e:
                if not quiet:
                    print(f"[compute_player_densities_dataframe] "
                          f"Skip nflId={r.get(id_col)} @ frame={frame_id}: {e}")

    all_player_df = pd.DataFrame(out_rows)

    # Put some nice, consistent column order (if present)
    preferred = [
        "frameId", id_col, name_col, jersey_col,
        "x", "y", "speed", "direction", is_off_col, dist_from_ball_col,
        "theta_rad", "orientation_bias_deg", density_out_col,
    ]
    cols_final = [c for c in preferred if c in all_player_df.columns] + \
                 [c for c in all_player_df.columns if c not in preferred]
    all_player_df = all_player_df[cols_final]

    return all_player_df, (X, Y)



def compute_player_densities_team_control(
    model,                         # PlayerInfluenceModel (required if recomputing)
    df: pd.DataFrame,
    *,
    # ---- which frames (pick ONE style) ----
    frame_id: Optional[int] = None,            # single frame
    frames: Optional[Iterable[int]] = None,    # explicit set/list of frames
    min_frame: Optional[int] = None,           # range mode (used if frames=None and frame_id=None)
    max_frame: Optional[int] = None,
    n_even: Optional[int] = None,              # optionally pick n evenly spaced frames from [min,max]

    # ---- grid/model ----
    grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (X, Y); else use model.grid or default

    # ---- influence → pitch-control knobs ----
    smooth_sigma: float = 2.0,
    pitch_k: float = 25.0,
    neutral_band: Tuple[float, float] = (0.45, 0.55),

    # ---- density caching/recompute ----
    use_cached_density: bool = True,
    force_recompute: bool = False,
    cache_back_to_df: bool = False,

    # ---- heading handling (consistent with your class) ----
    heading_col: str = "dir",      # prefer "dir"; falls back to "direction" if missing
    invert_heading: bool = False,  # set True ONLY if your data is known to be 180° flipped

    # ---- misc ----
    exclude_football: bool = True,
    verbose: bool = False,
):
    """
    Compute per-player Gaussian–Gamma influence and team pitch control for one OR many frames.

    Always returns multi-frame-shaped outputs; for a single frame you'll just have one key.

    Returns
    -------
    players_all : pd.DataFrame
        One row per (frameId, player) with columns like:
        ['frameId','nflId','displayName','jerseyNumber','x','y','speed','direction',
         'is_off','dist_from_football','density','sum_density', ...]
    offense_by_frame : Dict[int, pd.DataFrame]
        Smoothed offensive influence grid per frame (index=y, columns=x).
    defense_by_frame : Dict[int, pd.DataFrame]
        Smoothed defensive influence grid per frame (index=y, columns=x).
    pitch_by_frame : Dict[int, pd.DataFrame]
        Pitch control per frame (index=y, columns=x), with neutral band masked to NaN.
    grid_out : (X, Y)
        The meshgrid used.
    frames_used : List[int]
        The exact frameIds computed (in order).
    """
    # ------------------------ grid setup ------------------------
    if grid is not None:
        X, Y = grid
    elif hasattr(model, "grid") and getattr(model, "grid", None) is not None:
        try:
            X, Y = model.grid.X, model.grid.Y
        except AttributeError:
            X, Y = model.grid
    else:
        x_vals = np.linspace(0.0, 120.0, 200)
        y_vals = np.linspace(0.0, 53.3, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

    x_vals = X[0, :]
    y_vals = Y[:, 0]
    grid_out = (X, Y)

    # ------------------------ frame list ------------------------
    if frame_id is not None:
        frames_used = [int(frame_id)]
    elif frames is not None:
        frames_used = sorted({int(f) for f in frames})
    else:
        fmin = int(min_frame if min_frame is not None else df["frameId"].min())
        fmax = int(max_frame if max_frame is not None else df["frameId"].max())
        if n_even and n_even > 0:
            present = np.sort(df["frameId"].unique())
            targets = np.linspace(fmin, fmax, n_even)
            frames_used: List[int] = []
            for t in targets:
                nearest = int(present[np.argmin(np.abs(present - t))])
                if not frames_used or frames_used[-1] != nearest:
                    frames_used.append(nearest)
        else:
            frames_used = list(range(fmin, fmax + 1))

    # ------------------------ helpers ------------------------
    def _sigmoid(z, k): return 1.0 / (1.0 + np.exp(-k * z))
    def _safe_norm(A):
        mx = A.max()
        return A / mx if mx > 0 else A

    rows_out: List[dict] = []
    offense_by_frame: Dict[int, pd.DataFrame] = {}
    defense_by_frame: Dict[int, pd.DataFrame] = {}
    pitch_by_frame: Dict[int, pd.DataFrame] = {}

    # ------------------------ main loop ------------------------
    for fid in frames_used:
        frame_df = df[df["frameId"] == fid].copy()
        if frame_df.empty:
            if verbose:
                print(f"[pitch_control] frame {fid}: no rows; skipping")
            continue

        # filter out football & NaN is_off
        if exclude_football and "displayName" in frame_df.columns:
            frame_df = frame_df[frame_df["displayName"].str.lower() != "football"]
        if "is_off" in frame_df.columns:
            frame_df = frame_df[frame_df["is_off"].notna()]
        frame_df = frame_df.reset_index(drop=True)
        if frame_df.empty:
            if verbose:
                print(f"[pitch_control] frame {fid}: only football/NaN is_off rows; skipping")
            continue

        # -------- densities for this frame --------
        have_cache = ("density" in frame_df.columns) and frame_df["density"].notna().all()
        if use_cached_density and have_cache and not force_recompute:
            densities = np.stack(frame_df["density"].to_numpy(), axis=0)
        else:
            if model is None:
                raise ValueError(
                    f"Recompute requested but no `model` provided (frame={fid}). "
                    "Pass a PlayerInfluenceModel or set use_cached_density=True."
                )
            dens_list = []
            for _, r in frame_df.iterrows():
                x0, y0 = float(r["x"]), float(r["y"])
                pos = (x0, y0)

                # choose heading col robustly
                if heading_col in r and pd.notna(r[heading_col]):
                    dir_deg = float(r[heading_col])
                elif "direction" in r and pd.notna(r["direction"]):
                    dir_deg = float(r["direction"])
                else:
                    dir_deg = 0.0  # fallback; or raise if you prefer

                if invert_heading:
                    dir_deg = (dir_deg + 180.0) % 360.0

                spd = float(r["s"] if "s" in r else r.get("speed", 0.0))
                dball = float(r.get("dist_from_football", 0.0))

                pos_off = model.compute_offset(pos, dir_deg, spd)
                Z = model.base_distribution(
                    pos_xy=pos,
                    pos_off_xy=pos_off,
                    direction_deg=dir_deg,
                    speed=spd,
                    dist_from_ball=dball,
                )
                dens_list.append(Z)

            densities = np.stack(dens_list, axis=0)
            if cache_back_to_df:
                frame_df.loc[:, "density"] = dens_list

        # remove speckle
        densities = np.where(densities < 1e-3, 0.0, densities)

        # team splits
        is_off = frame_df["is_off"].to_numpy().astype(bool) if "is_off" in frame_df.columns else np.zeros(len(frame_df), bool)
        inf_off = densities[is_off].sum(axis=0) if is_off.any() else np.zeros_like(X)
        inf_def = densities[~is_off].sum(axis=0) if (~is_off).any() else np.zeros_like(X)

        # normalize + smooth
        inf_off = _safe_norm(inf_off)
        inf_def = _safe_norm(inf_def)
        if smooth_sigma and smooth_sigma > 0:
            inf_off = gaussian_filter(inf_off, sigma=smooth_sigma)
            inf_def = gaussian_filter(inf_def, sigma=smooth_sigma)

        # pitch control (defense→1.0); mask near 50/50
        pc = _sigmoid(inf_def - inf_off, k=pitch_k)
        lo, hi = neutral_band
        pc_masked = np.where((pc >= lo) & (pc <= hi), np.nan, pc)

        # per-player summary for this frame
        sum_density = densities.reshape(densities.shape[0], -1).sum(axis=1)
        frame_rows = frame_df.assign(frameId=fid, sum_density=sum_density).copy()

        # unify column names expected downstream
        if "s" in frame_rows.columns and "speed" not in frame_rows.columns:
            frame_rows = frame_rows.rename(columns={"s": "speed"})
        if heading_col in frame_rows.columns and "direction" not in frame_rows.columns:
            frame_rows = frame_rows.rename(columns={heading_col: "direction"})

        rows_out.append(frame_rows)

        # stash grids
        offense_by_frame[fid] = pd.DataFrame(inf_off, index=y_vals, columns=x_vals)
        defense_by_frame[fid] = pd.DataFrame(inf_def, index=y_vals, columns=x_vals)
        pitch_by_frame[fid]   = pd.DataFrame(pc_masked, index=y_vals, columns=x_vals)

        if verbose and (fid == frames_used[0] or fid == frames_used[-1] or fid % 10 == 0):
            print(f"[pitch_control] frame {fid}: players={len(frame_rows)}")

    players_all = pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
    return players_all, offense_by_frame, defense_by_frame, pitch_by_frame, grid_out, frames_used
