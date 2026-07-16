import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.certify_pre_atlas_v2 import (
    CertificationCache,
    SCHEMA_VERSION,
    _atomic_json,
    _canonical_hash,
    _finite_json,
    _frozen_configuration,
    _run_fingerprint,
    _settings,
    _sha256_files,
)


class CertificationV2Tests(unittest.TestCase):
    def test_strict_json_replaces_nonfinite_values_with_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.json"
            _atomic_json(
                path,
                {"nan": float("nan"), "infinity": float("inf"), "array": np.array([1.0])},
            )
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            self.assertEqual(json.loads(text)["nan"], None)
            self.assertEqual(_finite_json(complex(1.0, -2.0)), {"real": 1.0, "imag": -2.0})

    def test_frozen_configuration_has_self_consistent_v2_hash(self):
        frozen = _frozen_configuration(_settings())
        embedded = frozen.pop("config_sha256")
        self.assertEqual(frozen["schema_version"], SCHEMA_VERSION)
        self.assertEqual(embedded, _canonical_hash(frozen))
        self.assertEqual(len(frozen["alphas"]), 15)
        self.assertEqual(len(frozen["q_values"]), 33)
        self.assertEqual(len(frozen["channels"]), 5)
        self.assertEqual(frozen["expected_counts"]["unique_modes"], 120)

    def test_cache_rejects_corruption_and_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(
                code_sha256="code-a",
                config_sha256="config-a",
                dependency_sha256="deps-a",
                run_fingerprint="run-a",
            )
            cache = CertificationCache(root, enabled=True, **identity)
            cache.store("mode", "point", {"value": 3})
            self.assertEqual(cache.load("mode", "point"), {"value": 3})

            changed = CertificationCache(
                root,
                enabled=True,
                **{**identity, "code_sha256": "code-b"},
            )
            self.assertIsNone(changed.load("mode", "point"))

            path = cache._path("mode", "point")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["payload"]["value"] = 4
            path.write_text(json.dumps(envelope), encoding="utf-8")
            self.assertIsNone(cache.load("mode", "point"))

    def test_run_fingerprint_changes_with_every_identity_component(self):
        baseline = _run_fingerprint("code", "config", "deps")
        self.assertNotEqual(baseline, _run_fingerprint("other", "config", "deps"))
        self.assertNotEqual(baseline, _run_fingerprint("code", "other", "deps"))
        self.assertNotEqual(baseline, _run_fingerprint("code", "config", "other"))

    def test_source_hash_binds_path_and_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.py"
            second = root / "b.py"
            first.write_text("x = 1\n", encoding="utf-8")
            second.write_text("x = 1\n", encoding="utf-8")
            original = _sha256_files([first, second], root)
            second.write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(original, _sha256_files([first, second], root))


if __name__ == "__main__":
    unittest.main()
