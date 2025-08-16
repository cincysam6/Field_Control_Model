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
