"""
retails package
Provides:
- Path management utilities
- CSV loading/saving
- Model saving
- Filtered dataset caching
- Visualization utilities
"""

# -----------------------------
# Paths
# -----------------------------
from .paths import (
    get_path,
)

# -----------------------------
# Utils (data loading, saving, filtering)
# -----------------------------
from .utils import (
    load_csv,
    save_csv,
    save_model,
    build_filtered_filename,
    load_filtered_csv,
    load_data_filtered_by_date,
)

# -----------------------------
# Visualizations
# -----------------------------
from .visualizer import (
    apply_dark_theme,
    plot_time_series,
    plot_year_month_heatmap,
    plot_holiday_impact,
    plot_perishable_sales,
)

__all__ = [
    # paths
    "get_path",

    # utils
    "load_csv",
    "save_csv",
    "save_model",
    "build_filtered_filename",
    "load_filtered_csv",
    "load_data_filtered_by_date",

    # visualizer
    "apply_dark_theme",
    "plot_time_series",
    "plot_year_month_heatmap",
    "plot_holiday_impact",
    "plot_perishable_sales",
]
