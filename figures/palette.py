"""Frozen colour and typography contract for quantitative manuscript figures."""

from types import MappingProxyType


PALETTE = MappingProxyType(
    {
        "white": "#FFFFFF",
        "ink": "#34363A",
        "neutral_dark": "#6E7478",
        "neutral_mid": "#A1A8AC",
        "neutral_light": "#D6DCE0",
        "grid": "#EDF0F2",
        "band": "#F7F9FA",
        "blue": "#8CB7D3",
        "teal": "#91C3AF",
        "gold": "#E4C47F",
        "wine": "#D6A0B8",
    }
)


GROUP_STYLE = MappingProxyType(
    {
        "classic": (PALETTE["neutral_mid"], "o"),
        "same_task_sota": (PALETTE["blue"], "o"),
        "direct_mechanism": (PALETTE["gold"], "o"),
        "fcsr_fixed": (PALETTE["wine"], "o"),
        "fcsr_nested": (PALETTE["teal"], "o"),
    }
)


TARGET_COLORS = MappingProxyType(
    {
        "B5": PALETTE["blue"],
        "L": PALETTE["teal"],
        "BurB5": PALETTE["gold"],
        "BurL": PALETTE["wine"],
    }
)


FONT_FAMILY = "Times New Roman"
AXIS_LABEL_SIZE = 10.5
TICK_LABEL_SIZE = 9.0
LEGEND_SIZE = 8.0
PANEL_LABEL_SIZE = 10.0
SPINE_WIDTH = 0.30
GRID_WIDTH = 0.30
EXPORT_DPI = 600
OUTER_PAD_POINTS = 5.0
