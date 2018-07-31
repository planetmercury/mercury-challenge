import unittest
import sys
sys.path.append("../..")
from Baserate.main.historian import *
from ExpressScore.main.schema import JSONField as JF
from ExpressScore.main.schema import EventType


class TestHistorian(unittest.TestCase):

    def test_set_country(self):
        cc = "Iraq"
        h = Historian(country=cc)
        self.assertEqual(h.country, cc)

    def test_get_history(self):
        pass

    def test_get_history_from_path(self):
        pass

    def test_event_rate(self):
        pass


class TestMaHistorian(unittest.TestCase):

    def test_set_event_type(self):
        cc = "Egypt"
        h = MaHistorian(country=cc)
        self.assertEqual(h.event_type, EventType.MA)
