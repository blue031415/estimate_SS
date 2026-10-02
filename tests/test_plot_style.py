import unittest
from unittest.mock import patch

from matplotlib import rcParams

from utils.plot_style import _select_japanese_font, configure_matplotlib_font


class PlotStyleTest(unittest.TestCase):
    def test_preferred_installed_japanese_font_is_selected(self):
        selected = _select_japanese_font(["Meiryo", "Noto Sans JP"])

        self.assertEqual(selected, "Noto Sans JP")

    def test_matplotlib_uses_the_selected_font(self):
        fonts = [type("Font", (), {"name": "BIZ UDGothic"})()]
        original_family = rcParams["font.family"]
        original_unicode_minus = rcParams["axes.unicode_minus"]

        try:
            with patch("utils.plot_style.font_manager.fontManager.ttflist", fonts):
                selected = configure_matplotlib_font()

            self.assertEqual(selected, "BIZ UDGothic")
            self.assertEqual(rcParams["font.family"], ["BIZ UDGothic"])
            self.assertFalse(rcParams["axes.unicode_minus"])
        finally:
            rcParams["font.family"] = original_family
            rcParams["axes.unicode_minus"] = original_unicode_minus


if __name__ == "__main__":
    unittest.main()
