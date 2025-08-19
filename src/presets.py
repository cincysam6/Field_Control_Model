# --------------------------------------------------------------------------- #
# Default Configuration for PlayerInfluenceModel
# --------------------------------------------------------------------------- #

default_kwargs: dict = dict(
    # ---------------- Grid resolution / field dimensions ---------------- #
    grid_x_res=200,      # number of points in X direction (field length)
    grid_y_res=100,      # number of points in Y direction (field width)
    field_x_max=120.0,   # field length in yards
    field_y_max=53.3,    # field width in yards

    # ---------------- Orientation bias ---------------------------------- #
    orientation_bias_deg=0.0,

    # ---------------- Gaussian component -------------------------------- #
    gaussian_scale_factor=0.7,
    # Scales the Gaussian ellipse size. Larger → more diffuse influence.

    # ---------------- Gamma component ----------------------------------- #
    alpha_gamma=11.0,
    # Shape parameter of gamma distribution; controls forward density shape.
    beta_min=1.0,
    beta_max=20.0,
    # Dynamic beta depends on speed; clipped between [beta_min, beta_max].
    gamma_midpoint=15.0,
    # Speed (yards/s) where beta ramp is centered; affects how reach scales.
    gamma_scale_factor=0.8,
    # Multiplicative shrink/expand factor for gamma influence footprint.
    max_forward_distance=20.0,
    # Cutoff distance (yards) beyond which gamma influence fades.
    forward_decay_factor=1.0,
    # Controls exponential decay after the forward cutoff. Higher = slower fade.

    # ---------------- Angular cone filtering ----------------------------- #
    angle_limit_min=15.0,
    # Cone half-angle (deg) at very high speed (narrow).
    angle_limit_max=45.0,
    # Cone half-angle (deg) at very low speed (wide).
    angle_decay_factor=2.0,
    # How fast cone narrows with speed. Lower = cone shrinks more quickly.

    # ---------------- Mixture weights (Gaussian vs Gamma) ---------------- #
    w_gaussian_min=0.2,
    # Minimum Gaussian weight (at very high speed).
    w_gaussian_max=1.0,
    # Maximum Gaussian weight (at near-zero speed).
    gaussian_midpoint=4.0,
    # Logistic midpoint: speed (yd/s) where weights are ~50/50.
    gaussian_steepness=2.0,
    # Logistic steepness: how sharply weights transition around midpoint.
)


###############


triangular_kwargs = dict(
    # Grid
    grid_x_res=200,
    grid_y_res=100,
    field_x_max=120.0,
    field_y_max=53.3,

    # Gaussian
    gaussian_scale_factor=0.8,    # rounder, slightly larger low-speed blobs

    # Gamma
    alpha_gamma=6.0,              # less pointy, more smooth elongation
    beta_min=4.0,                 # not too short-tailed at low v
    beta_max=12.0,                # not too long-tailed at high v
    gamma_midpoint=12.0,          # elongation centered fairly high
    gamma_scale_factor=0.9,
    max_forward_distance=12.0,    # short forward cutoff
    forward_decay_factor=1.0,

    # Angular cone
    angle_limit_min=20.0,         # fairly wide even at high speeds
    angle_limit_max=60.0,         # very wide at low speeds (round blobs)
    angle_decay_factor=2.0,

    # Mixture weights
    w_gaussian_min=0.3,           # Gaussian never disappears
    w_gaussian_max=1.0,
    gaussian_midpoint=3.0,        # forward component starts early
    gaussian_steepness=1.5        # smooth logistic transition
)


