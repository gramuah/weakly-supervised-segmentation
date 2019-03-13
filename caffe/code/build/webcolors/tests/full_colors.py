"""
Test case which exercises webcolors' conversion functions across all
16,777,216 possible hexadecimal values and all 16,777,216 possible
integer rgb() triplet values.

You should not ever need to run this test; it is not part of the
normal unit-test suite, and is used only as a final check when
preparing a new release of webcolors.

Because it generates each of the nearly 17 million color values
multiple times, this test case takes some time to run and consumes
most or all available CPU while running. As a consolation, it is
somewhat efficient with respect to memory.

Due to the inherent imprecision of floating-point percentage values,
and the fact that the legal (with respect to the CSS standards) set of
percentage rgb() triplets is uncountably infinite, percentage rgb()
triplets are not exhaustively tested here, and the normal test suite
is used to ensure correctness of the conversion functions for those
values.

The only test performed here for percentage rgb() triplets is to
ensure that converting an integer rgb() triplet to percentage and back
returns the original integer values, for consistency.

"""

import unittest

import webcolors

try:
    # On Python 3, the built-in zip() is identical to the Python 2
    # itertools.izip(), and Python 3's itertools does not define an
    # izip() as a result.
    from itertools import izip as zip
except ImportError:
    pass

try:
    # Similarly, Python 3's range() does what Python 2's xrange() did,
    # and so Python 3 does not have xrange()
    xrange(1)
except NameError:
    xrange = range


def hex_colors():
    HEX_TEMPLATE = u"#%06x"
    for i in xrange(16777217):
        yield HEX_TEMPLATE % i


def int_colors():
    red_counter = tuple(range(256))
    green_counter = tuple(range(256))
    blue_counter = tuple(range(256))
    for red_value in red_counter:
        for green_value in green_counter:
            for blue_value in blue_counter:
                yield (red_value, green_value, blue_value)


class FullColorTest(unittest.TestCase):
    def test_full_colors(self):
        for hex_color, int_triplet in zip(hex_colors(), int_colors()):
            self.assertEqual(int_triplet,
                             webcolors.hex_to_rgb(hex_color))
            self.assertEqual(hex_color,
                             webcolors.rgb_to_hex(int_triplet))

    def test_triplet_conversion(self):
        for int_triplet in int_colors():
            self.assertEqual(int_triplet,
                             webcolors.rgb_percent_to_rgb(
                                 webcolors.rgb_to_rgb_percent(int_triplet)))


if __name__ == '__main__':
    unittest.main()
