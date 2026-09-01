"""CPU tests for the pure parts of the CMIP6 download (no network)."""

import unittest

import numpy as np

from data.download_cmip6 import CMIP6_MAP, G, cmip6_spec, coarse_grid


class MappingTests(unittest.TestCase):
    def test_all_20var_channels_have_a_mapping(self):
        surface = ["2m_temperature", "10m_u_component_of_wind",
                   "10m_v_component_of_wind", "mean_sea_level_pressure",
                   "total_column_water_vapour"]
        plev = ["geopotential", "temperature", "u_component_of_wind",
                "v_component_of_wind", "specific_humidity"]
        for name in surface:
            var, p, f = cmip6_spec({"name": name, "level": None})
            self.assertIsNone(p)
        for name in plev:
            for lvl in (500, 700, 850):
                var, p, f = cmip6_spec({"name": name, "level": lvl})
                self.assertEqual(p, lvl * 100.0)   # hPa -> Pa

    def test_geopotential_unit_conversion(self):
        var, p, factor = cmip6_spec({"name": "geopotential", "level": 500})
        self.assertEqual(var, "zg")
        self.assertEqual(factor, G)
        # everything else passes through unchanged
        for name, (v, f) in CMIP6_MAP.items():
            if name != "geopotential":
                self.assertEqual(f, 1.0, name)

    def test_unknown_variable_rejected(self):
        with self.assertRaises(KeyError):
            cmip6_spec({"name": "sea_surface_temperature", "level": None})


class CoarseGridTests(unittest.TestCase):
    def test_block_means_and_trim(self):
        lat = np.linspace(60, -60, 481)     # WB2 band: 481 rows, not /4
        lon = np.arange(0, 360, 0.25)       # 1440 cols
        lat_c, lon_c = coarse_grid(lat, lon, 4)
        self.assertEqual(len(lat_c), 120)   # 480 kept, 1 trimmed
        self.assertEqual(len(lon_c), 360)
        self.assertAlmostEqual(float(lat_c[0]), float(lat[:4].mean()), places=10)
        self.assertAlmostEqual(float(lon_c[-1]), float(lon[1436:1440].mean()), places=10)

    def test_ratio_one_is_identity(self):
        lat = np.linspace(-10, 10, 8)
        lat_c, _ = coarse_grid(lat, lat, 1)
        self.assertTrue(np.allclose(lat_c, lat))


if __name__ == "__main__":
    unittest.main()
