"""Tests minimaux (unittest, sans dépendance pytest)."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
from PIL import Image


class TestSmoke(unittest.TestCase):
    def setUp(self):
        fd, self.png = tempfile.mkstemp(suffix=".png")
        import os

        os.close(fd)
        Image.fromarray(np.random.randint(0, 255, (24, 24), dtype=np.uint8)).save(self.png)

    def tearDown(self):
        import os

        os.unlink(self.png)

    def test_patch_reconstruction(self):
        from backend.Tratement_Image import patch

        o = patch(
            self.png,
            B=8,
            ratio=0.4,
            method="omp",
            dictionary_type="dct",
            max_patches=4,
            max_iter=15,
            epsilon=1e-5,
            seed=1,
            as_dict=True,
        )
        self.assertIn("coherence_mutuelle_cours", o)

    def test_cosamp_s_fixe(self):
        from backend.Tratement_Image import patch

        o = patch(
            self.png,
            B=8,
            ratio=0.5,
            method="cosamp",
            s_cosamp=4,
            s_cosamp_auto=False,
            max_patches=2,
            max_iter=10,
            epsilon=1e-4,
            seed=0,
            as_dict=True,
        )
        self.assertEqual(o.get("cosamp_s_mode"), "fixe")
        self.assertEqual(o.get("cosamp_s"), 4)

    def test_cosamp_s_auto(self):
        from backend.Tratement_Image import patch

        o = patch(
            self.png,
            B=8,
            ratio=0.5,
            method="cosamp",
            s_cosamp=6,
            s_cosamp_auto=True,
            max_patches=4,
            max_iter=12,
            epsilon=1e-4,
            seed=0,
            as_dict=True,
        )
        self.assertEqual(o.get("cosamp_s_mode"), "estime_omp")
        self.assertIsInstance(o.get("cosamp_s"), int)

    def test_run_pipeline(self):
        from main import run_pipeline, setupParam

        p = setupParam(
            self.png,
            8,
            0.35,
            ["omp"],
            "dct",
            patch_params={"max_patches": 3},
            seed=2,
        )
        out = run_pipeline(p, etapes=("reconstruct",))
        self.assertIn("omp", out["reconstruction"]["metrics"])


if __name__ == "__main__":
    unittest.main()
