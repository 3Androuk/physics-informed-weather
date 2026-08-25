"""Unit tests for the eval.figure_channels selection helper."""

import unittest

from utils import figure_channels


def _cfg(selection="ABSENT"):
    cfg = {"data": {"variables": [
        {"name": "2m_temperature", "level": None},
        {"name": "geopotential", "level": 500},
        {"name": "specific_humidity", "level": 850},
    ]}}
    if selection != "ABSENT":
        cfg["eval"] = {"figure_channels": selection}
    return cfg


class FigureChannelsTests(unittest.TestCase):
    def test_absent_and_null_select_all_channels(self):
        self.assertEqual(figure_channels(_cfg()), [0, 1, 2])
        self.assertEqual(figure_channels(_cfg(None)), [0, 1, 2])

    def test_labels_indices_and_mixing(self):
        self.assertEqual(figure_channels(_cfg(["t2m", "q850"])), [0, 2])
        self.assertEqual(figure_channels(_cfg([2, 0])), [2, 0])
        self.assertEqual(figure_channels(_cfg(["z500", 0])), [1, 0])

    def test_single_variable_legacy_config(self):
        cfg = {"data": {"variable": "geopotential", "level": 500}}
        self.assertEqual(figure_channels(cfg), [0])

    def test_unknown_label_and_bad_index_raise(self):
        with self.assertRaises(ValueError):
            figure_channels(_cfg(["not_a_channel"]))
        with self.assertRaises(ValueError):
            figure_channels(_cfg([3]))


if __name__ == "__main__":
    unittest.main()
