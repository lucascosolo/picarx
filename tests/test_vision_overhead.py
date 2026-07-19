import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import vision_basic as vb  # noqa: E402


class PickOverheadTest(unittest.TestCase):
    """pick_overhead is pure geometry (no cv2/camera), so it's exercised
    directly. Boxes are (area_ratio, y_center_frac, y_bottom_frac)."""

    def test_none_when_no_boxes(self):
        self.assertIsNone(vb.pick_overhead([]))

    def test_high_and_big_mass_qualifies(self):
        # Big, centered high in the frame, floor open below -> an overhang.
        out = vb.pick_overhead([(0.40, 0.30, 0.60)])
        self.assertIsNotNone(out)
        self.assertEqual(out["area_ratio"], 0.40)
        self.assertEqual(out["y_center_frac"], 0.30)

    def test_full_height_wall_rejected(self):
        # A big mass running all the way to the floor is a wall, not an
        # overhang - its bottom exceeds OVERHEAD_MAX_BOTTOM.
        self.assertIsNone(vb.pick_overhead([(0.70, 0.50, 0.99)]))

    def test_low_mass_rejected(self):
        # Big, but sitting LOW in the frame (a floor-level object the
        # ultrasonic would see itself) - not overhead.
        self.assertIsNone(vb.pick_overhead([(0.50, 0.80, 0.95)]))

    def test_small_high_speck_rejected(self):
        # High but too small to be imminent.
        self.assertIsNone(vb.pick_overhead([(0.05, 0.20, 0.30)]))

    def test_picks_largest_qualifying(self):
        out = vb.pick_overhead([
            (0.32, 0.40, 0.60),   # qualifies
            (0.55, 0.35, 0.55),   # qualifies, bigger -> chosen
            (0.90, 0.60, 0.99),   # full-height wall, rejected
        ])
        self.assertEqual(out["area_ratio"], 0.55)


class EdgeTruncationTest(unittest.TestCase):
    """edge_truncation is pure geometry: which borders cut a box off, so
    downstream knows it's only seeing PART of the object."""
    W, H = 320, 240

    def _edges(self, x, y, w, h):
        return vb.edge_truncation(x, y, w, h, self.W, self.H)

    def test_fully_in_view_is_not_truncated(self):
        self.assertEqual(self._edges(100, 80, 60, 60), [])

    def test_left_border_cutoff(self):
        self.assertEqual(self._edges(0, 80, 60, 60), ["left"])

    def test_right_border_cutoff(self):
        self.assertEqual(self._edges(280, 80, 40, 60), ["right"])

    def test_bottom_cutoff_partial_person(self):
        # A person whose legs run off the bottom - still a person.
        self.assertEqual(self._edges(120, 100, 80, 140), ["bottom"])

    def test_top_half_of_head_out_of_frame(self):
        # Box pinned to the top edge: the top of the head is cut off.
        self.assertEqual(self._edges(120, 0, 80, 120), ["top"])

    def test_corner_reports_both_sides(self):
        self.assertEqual(self._edges(0, 0, 80, 80), ["left", "top"])

    def test_within_margin_counts_as_touching(self):
        # A couple of pixels off the border still reads as cut off.
        self.assertEqual(self._edges(2, 80, 60, 60), ["left"])


if __name__ == "__main__":
    unittest.main()
