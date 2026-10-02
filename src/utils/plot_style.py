"""Shared Matplotlib settings for figures containing Japanese text."""

from matplotlib import font_manager, rcParams


JAPANESE_FONT_CANDIDATES = (
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "BIZ UDGothic",
    "Yu Gothic",
    "Meiryo",
    "IPAexGothic",
    "IPAGothic",
    "MS Gothic",
)


def _select_japanese_font(available_font_names):
    available = set(available_font_names)
    return next(
        (name for name in JAPANESE_FONT_CANDIDATES if name in available),
        None,
    )


def configure_matplotlib_font():
    """Use an installed Japanese font for every subsequently created figure."""
    font_name = _select_japanese_font(
        font.name for font in font_manager.fontManager.ttflist
    )
    if font_name is not None:
        rcParams["font.family"] = [font_name]
    rcParams["axes.unicode_minus"] = False
    return font_name
