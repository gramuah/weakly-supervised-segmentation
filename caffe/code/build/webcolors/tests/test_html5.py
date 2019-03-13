import unittest

import webcolors


class HTML5Tests(unittest.TestCase):
    """
    Test functions which implement the HTML5 color algorithms.
    """
    def test_parse_simple_color(self):
        """
        Test implementation of the HTML5 simple color parsing
        algorithm.
        """
        test_pairs = ((u'#ffffff', (255, 255, 255)),
                      (u'#000080', (0, 0, 128)),
                      (u'#daa520', (218, 165, 32)))
        for pair in test_pairs:
            self.assertEqual(pair[1],
                             webcolors.html5_parse_simple_color(pair[0]))

    def test_parse_simple_color_error(self):
        """
        Test error conditions of the HTML5 simple color parsing
        algorithm.
        """
        test_values = (u'0099cc',
                       u'#09c',
                       u'#0000',
                       u'#0000000',
                       u'#0000gg',
                       u'#000000'.encode('ascii'))
        for value in test_values:
            self.assertRaises(ValueError,
                              webcolors.html5_parse_simple_color,
                              value)

    def test_serialize_simple_color(self):
        """
        Test implementation of the HTML5 simple color serialization
        algorithm.
        """
        test_pairs = (((0, 0, 0), u'#000000'),
                      ((0, 0, 128), u'#000080'),
                      ((218, 165, 32), u'#daa520'))
        for pair in test_pairs:
            self.assertEqual(pair[1],
                             webcolors.html5_serialize_simple_color(pair[0]))

    def test_parse_legacy_color(self):
        """
        Test implementation of the HTML5 legacy color parsing
        algorithm.
        """
        # One of these is the famous "chucknorris" value. Another is a
        # CSS 2 system color. The final two are randomly-generated but
        # believable junk strings. Correct output values obtained
        # manually.
        test_pairs = ((u'chucknorris', (192, 0, 0)),
                      (u'Window', (0, 13, 0)),
                      (u'RE|SXLuAse', (224, 0, 224)),
                      (u'+=@FnnWL!Yb}5Dk', (0, 0, 176)))
        for pair in test_pairs:
            self.assertEqual(pair[1],
                             webcolors.html5_parse_legacy_color(pair[0]))

    def test_parse_legacy_color_names(self):
        """
        Test the HTML5 legacy color parsing of SVG/CSS3 color names.
        """
        for name in webcolors.CSS3_NAMES_TO_HEX.keys():
            self.assertEqual(webcolors.name_to_rgb(name),
                             webcolors.html5_parse_legacy_color(name))

    def test_parse_legacy_color_hex(self):
        """
        Test the HTML5 legacy color parsing of three- and six-digit
        hexadecimal color values.
        """
        test_values = (u'#000',
                       u'#000000',
                       u'#fff',
                       u'#ffffff',
                       u'#000080')
        for value in test_values:
            self.assertEqual(webcolors.hex_to_rgb(value),
                             webcolors.html5_parse_legacy_color(value))

    def test_parse_legacy_color_error(self):
        """
        Test error conditions of the HTML5 legacy color parsing algorithm.
        """
        test_values = (u'#000000'.encode('ascii'),
                       u"transparent",
                       u'')
        for value in test_values:
            self.assertRaises(ValueError,
                              webcolors.html5_parse_legacy_color,
                              value)
