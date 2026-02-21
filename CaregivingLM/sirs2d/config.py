"""
Global configuration parameters for SIRS 2D system.
"""

import numpy as np

# Global joint limits (degrees)
Q1_RANGE = (-150.0, 150.0)  # Joint 1 global limits
Q2_RANGE = (-50.0, 180.0)   # Joint 2 global limits

# Box sampling parameters
BOX_SIZE_FRACTION_MIN = 0.35  # Minimum box size as fraction of global range
BOX_SIZE_FRACTION_MAX = 0.8   # Maximum box size as fraction of global range

# Bump parameters
NUM_BUMPS_MIN = 1
NUM_BUMPS_MAX = 4
EDGE_BIAS_BETA_ALPHA = 0.2  # Beta distribution α parameter for edge-biased sampling (α < 1 = edge bias, α > 1 = center bias)
ENABLE_ROTATION = True  # Enable rotated (non-axis-aligned) Gaussians for joint coupling

# Alpha (bump strength) parameters - log-normal distribution
ALPHA_LOGNORMAL_MEAN = 4.0   # ln(alpha) mean
ALPHA_LOGNORMAL_SIGMA = 0.6   # ln(alpha) std deviation

# Lengthscale parameters
LENGTHSCALE_BOX_FRACTION = 0.15  # Lengthscale as fraction of box width
LENGTHSCALE_LOGNORMAL_SIGMA = 0.4  # Log-normal variation around mean
LENGTHSCALE_MIN_FRACTION = 0.05  # Minimum lengthscale as fraction of box width
LENGTHSCALE_MAX_FRACTION = 0.5   # Maximum lengthscale as fraction of box width

# Grid resolution for evaluation
GRID_RESOLUTION_MIN = 300
GRID_RESOLUTION_MAX = 400

# Visualization parameters
COLORMAP = "viridis"
FEASIBLE_COLOR = "green"
INFEASIBLE_COLOR = "lightgray"
CONTOUR_COLOR = "black"
CONTOUR_LINEWIDTH = 2.5
BOX_COLOR = "blue"
BOX_LINESTYLE = "--"
BOX_LINEWIDTH = 1.5
BUMP_CENTER_MARKER = "x"
BUMP_CENTER_COLOR = "red"
BUMP_CENTER_SIZE = 100
BUMP_ELLIPSE_COLOR = "red"
BUMP_ELLIPSE_ALPHA = 0.3
BUMP_ELLIPSE_LINEWIDTH = 1.5

# Calibration parameters
CALIBRATION_MAX_ITERATIONS = 50
CALIBRATION_TOLERANCE = 0.05  # ±5% of target fraction
CALIBRATION_BINARY_SEARCH_FACTOR = 2.0

# Random seed for reproducibility (None for random)
DEFAULT_RANDOM_SEED = 42

# Smooth corner parameters
SMOOTHING_K = 0.0  # Default smoothing radius (0 = sharp corners, >0 = smooth)
SMOOTHING_K_AUTO_FRACTION = 0.1  # Auto-tuning: k = 10% of smallest box dimension
