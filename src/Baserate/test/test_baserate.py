import unittest
import sys
sys.path.append("../..")
from Baserate.main.baserate import *
from ExpressScore.main.schema import JSONField as JF
from ExpressScore.main.schema import EventType


class TestBaserate(unittest.TestCase):

    def test_country(self):
        """
        Tests setting of country attribute
        :return:
        """

        cc = "Qatar"
        b = Baserate(country=cc)
        self.assertEqual(cc, b.country)

class TestMaBaserate(unittest.TestCase):

    def test_event_type(self):

        cc = "Qatar"
        b = MaBaserate(country=cc)
        self.assertEqual(cc, b.country)
        self.assertEqual(b.historian.event_type, EventType.MA)
        self.assertEqual(b.historian.country, cc)
