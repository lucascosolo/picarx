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


if __name__ == "__main__":
    unittest.main()
