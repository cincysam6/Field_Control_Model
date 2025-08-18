import numpy as np
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd


# Function to compute distance from the football for a given frame
def compute_distances_by_frame(frame_data):
    # Isolate the football's position within the frame
    football_row = frame_data[frame_data['displayName'] == 'football']
    if football_row.empty:
        # If no football is present in the frame, return NaN for distances
        frame_data['dist_from_football'] = np.nan
    else:
        football_x = football_row['x'].values[0]
        football_y = football_row['y'].values[0]
        # Compute the distance for all players in the frame
        frame_data['dist_from_football'] = np.sqrt((frame_data['x'] - football_x) ** 2 +
                                                   (frame_data['y'] - football_y) ** 2)
    return frame_data


# -------------------------- tiny helpers ------------------------------------
def pick_panel_kwargs(
    panel_params: List[Dict[str, Any]],
    *,
    title: Optional[str] = None,
    index: int = 0,
    updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience: select a panel's parameter dict and (optionally) apply updates,
    yielding a ready-to-use `model_kwargs` dictionary.

    Examples
    --------
    >>> _, panels, _ = diagnostic_multiples(..., verbose=True)
    >>> mk = pick_panel_kwargs(panels, title="Gamma reach ↑", updates={"alpha_gamma": 12.0})
    >>> model = PlayerInfluenceModel(**mk)
    """
    if title is not None:
        matches = [p for p in panel_params if p.get("title") == title]
        if not matches:
            raise ValueError(f"No panel titled '{title}' found.")
        params = dict(matches[0]["params"])
    else:
        params = dict(panel_params[int(index)]["params"])

    # remove non-model keys (like 'speed') if present
    params.pop("speed", None)

    if updates:
        params.update(updates)
    return params


def update_kwargs(base_kwargs: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
    """
    Copy-and-update helper: start from an existing kwargs dict and return a modified copy.
    """
    out = dict(base_kwargs)
    out.update(updates)
    return out



def heading_alignment_check(
    df: pd.DataFrame,
    pid: Union[int, str],
    *,
    frame_col: str = "frameId",
    id_col: str = "nflId",
    x_col: str = "x",
    y_col: str = "y",
    heading_col: str = "dir",   # 0° = north (+Y), clockwise increasing
    min_displacement: float = 1e-6,   # guard against 0 move between frames
    make_plot: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Check how well the recorded heading (e.g., 'dir') aligns with actual movement
    between consecutive frames for ONE player.

    Method
    ------
    For each consecutive pair of frames, compute:
      - displacement Δ = (dx, dy) from (x,y) differences
      - heading unit vector h from tracking degrees using θ = radians(90 - deg)
      - cosine alignment cosθ = (h · Δ) / (||h||·||Δ||), in [-1, 1]

    Interpretation
    --------------
      ~ +1 : heading points in the same direction the player actually moved
      ~  0 : heading is orthogonal to movement
      ~ -1 : heading is opposite the movement (possible 180° flip)
    We summarize over frames with mean, median, std, and share of negatives.

    Parameters
    ----------
    df : DataFrame containing at least [frame_col, id_col, x_col, y_col, heading_col]
          for the player of interest.
    pid : Player id value in df[id_col].
    frame_col, id_col, x_col, y_col, heading_col : column names.
    min_displacement : small epsilon to avoid 0-length displacement division.
    make_plot : if True, produces a simple cosθ vs frame plot.

    Returns
    -------
    per_frame : DataFrame with columns:
        [frame_col, x, y, heading_deg, dx, dy, disp, hx, hy, cos_align]
    summary : dict with keys:
        {'mean', 'median', 'std', 'share_negative', 'n_pairs', 'player_id'}

    Notes
    -----
    - Rows with NaN in any of the needed columns are dropped prior to calc.
    - First frame has no previous frame → cos_align is NaN for that row.
    """
    g = (
        df.loc[df[id_col] == pid, [frame_col, x_col, y_col, heading_col]]
          .dropna(subset=[frame_col, x_col, y_col, heading_col])
          .sort_values(frame_col)
          .copy()
    )
    if g.empty:
        raise ValueError(f"No rows found for {id_col}={pid}")

    # Displacements between consecutive frames
    g["dx"] = g[x_col].diff()
    g["dy"] = g[y_col].diff()
    g["disp"] = np.sqrt(g["dx"]**2 + g["dy"]**2).clip(lower=min_displacement)

    # Heading unit vector from tracking degrees (0°=north/up, clockwise)
    theta = np.deg2rad((90.0 - g[heading_col]) % 360.0)
    g["hx"] = np.cos(theta)
    g["hy"] = np.sin(theta)

    # Cosine alignment
    # (h · Δ) / (||h|| * ||Δ||); ||h|| is ~1, but keep explicit for clarity
    hnorm = np.sqrt(g["hx"]**2 + g["hy"]**2).replace(0.0, np.nan)
    dot = g["hx"] * g["dx"] + g["hy"] * g["dy"]
    g["cos_align"] = dot / (hnorm * g["disp"])

    # Build a tidy per-frame view
    per_frame = g.rename(
        columns={
            x_col: "x",
            y_col: "y",
            heading_col: "heading_deg"
        }
    )[[frame_col, "x", "y", "heading_deg", "dx", "dy", "disp", "hx", "hy", "cos_align"]]

    # Summary stats (ignore first NaN)
    valid = per_frame["cos_align"].dropna()
    summary = {
        "player_id": pid,
        "n_pairs": int(valid.shape[0]),
        "mean": float(valid.mean()) if not valid.empty else np.nan,
        "median": float(valid.median()) if not valid.empty else np.nan,
        "std": float(valid.std(ddof=1)) if valid.shape[0] > 1 else np.nan,
        "share_negative": float((valid < 0).mean()) if not valid.empty else np.nan,
    }

    # Optional quick plot
    if make_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(per_frame[frame_col], per_frame["cos_align"], marker="o", lw=1)
        ax.axhline(0.0, color="k", lw=1, alpha=0.4)
        ax.set_title(f"Heading vs Movement • {id_col}={pid} (mean cos≈{summary['mean']:.2f})")
        ax.set_xlabel(frame_col)
        ax.set_ylabel("cosine alignment")
        ax.grid(alpha=0.25)
        plt.show()

    return per_frame, summary


def heading_alignment_summary(
    df: pd.DataFrame,
    pids: Sequence[Union[int, str]],
    *,
    frame_col: str = "frameId",
    id_col: str = "nflId",
    x_col: str = "x",
    y_col: str = "y",
    heading_col: str = "dir",
    min_displacement: float = 1e-6,
) -> pd.DataFrame:
    """
    Run `heading_alignment_check` for multiple players and return a tidy summary table.

    Returns a DataFrame with one row per player id and columns:
      [player_id, n_pairs, mean, median, std, share_negative]
    """
    rows = []
    for pid in pids:
        try:
            _, s = heading_alignment_check(
                df, pid,
                frame_col=frame_col, id_col=id_col,
                x_col=x_col, y_col=y_col, heading_col=heading_col,
                min_displacement=min_displacement, make_plot=False
            )
            rows.append(s)
        except Exception as e:
            rows.append({
                "player_id": pid, "n_pairs": 0,
                "mean": np.nan, "median": np.nan, "std": np.nan, "share_negative": np.nan,
                "error": str(e),
            })
    out = pd.DataFrame(rows)
    # helpful sort: worst to best
    if "mean" in out:
        out = out.sort_values("mean", ascending=True, na_position="last").reset_index(drop=True)
    return out
