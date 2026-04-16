import unittest
from qAttitude import wrap360, deg2rad, rad2deg, trend_plunge_to_lmn, lmn_to_trend_plunge
import math

class TestQAttitudeLogic(unittest.TestCase):

    def test_wrap360(self):
        self.assertEqual(wrap360(361), 1)
        self.assertEqual(wrap360(-1), 359)
        self.assertEqual(wrap360(0), 0)
        self.assertEqual(wrap360(360), 0)

    def test_deg2rad_rad2deg(self):
        self.assertAlmostEqual(deg2rad(180), math.pi)
        self.assertAlmostEqual(rad2deg(math.pi), 180)

    def test_trend_plunge_to_lmn_roundtrip(self):
        # Test a few points
        test_points = [(0, 0), (90, 45), (180, 90), (270, 30)]
        for trend, plunge in test_points:
            l, m, n = trend_plunge_to_lmn(trend, plunge)
            t2, p2 = lmn_to_trend_plunge(l, m, n)
            self.assertAlmostEqual(trend, t2, places=5)
            self.assertAlmostEqual(plunge, p2, places=5)

if __name__ == '__main__':
    unittest.main()
