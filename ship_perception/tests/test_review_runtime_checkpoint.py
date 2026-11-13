import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ship_perception.perception import review_runtime


class ReviewRuntimeCheckpoint(unittest.TestCase):
    def test_heartbeats_reuse_frozen_geometry_and_keep_current_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            monitor = review_runtime.ReviewRunMonitor(Path(temp))
            objects = [{"opening_identity": "船舱", "corner_z_m": [None] * 4}]
            original = review_runtime._json_bytes
            with patch.object(review_runtime, "_json_bytes", wraps=original) as encoded:
                monitor.checkpoint("CORNER_HEIGHT_OUTPUT", review_objects=objects)
                objects[0]["corner_z_m"][0] = 1.0
                tasks = review_runtime.MeasurementTasks("input", monitor)
                tasks.flush({"normal_calls": 5})
                tasks.flush({"normal_calls": 6})
                value = json.loads((monitor.folder / "checkpoint.json").read_bytes())
                self.assertEqual(
                    sum(isinstance(call.args[0], list) for call in encoded.call_args_list), 1
                )
            self.assertEqual(value["stage"], "CORNER_HEIGHT_OUTPUT")
            self.assertEqual(value["cache_state"], {"normal_calls": 6})
            self.assertEqual(value["review_objects"][0]["corner_z_m"], [None] * 4)
            monitor.checkpoint(review_objects=[])
            value = json.loads((monitor.folder / "checkpoint.json").read_bytes())
            self.assertEqual(value["review_objects"], [])
