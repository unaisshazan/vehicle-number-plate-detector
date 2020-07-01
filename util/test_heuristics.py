from unittest import TestCase

from util.heuristics import *


class TestHeuristics(TestCase):

    def test_join_separated(self):
        bands = [(141, 221, 176, 215), (141, 221, 134, 171), (141, 221, 215, 246), (21, 140, 344, 399), (21, 140, 399, 444),
         (21, 140, 186, 273), (220, 252, 436, 487), (220, 252, 487, 976), (42, 59, 171, 225), (255, 282, 42, 75)]

        bands = [(220, 252, 436, 487), (220, 252, 487, 976), (42, 59, 171, 225), (255, 282, 42, 75)]

        actual = join_separated(bands)
        expected = [(255, 282, 42, 75), (42, 59, 171, 225), (220, 252, 436, 976)]

        self.assertEqual(actual, expected)


