import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from deadlock_state import has_dependency_cycle, interpret_prediction


class DeadlockStateTests(unittest.TestCase):
    def test_safe_path(self):
        graph = {"edges": [{"source": "t1", "target": "l1", "type": "waits_for"}]}
        self.assertFalse(has_dependency_cycle(graph))
        self.assertEqual(interpret_prediction(graph, {"state": "safe"})["state"], "safe")

    def test_model_deadlocked_before_observed_cycle_is_warning(self):
        graph = {"edges": [{"source": "t1", "target": "l1", "type": "waits_for"}]}
        result = interpret_prediction(graph, {"state": "deadlocked"})
        self.assertEqual(result["state"], "pre_deadlock")
        self.assertEqual(result["model_state"], "deadlocked")
        self.assertFalse(result["observed_cycle"])

    def test_confirmed_cycle_overrides_model_safe(self):
        graph = {"edges": [
            {"source": "t1", "target": "l1", "type": "waits_for"},
            {"source": "l1", "target": "t2", "type": "owned_by"},
            {"source": "t2", "target": "l2", "type": "waits_for"},
            {"source": "l2", "target": "t1", "type": "owned_by"},
        ]}
        result = interpret_prediction(graph, {"state": "safe"})
        self.assertEqual(result["state"], "deadlocked")
        self.assertTrue(result["observed_cycle"])

    def test_acyclic_contention_not_deadlocked(self):
        graph = {"edges": [
            {"source": "t1", "target": "l1", "type": "waits_for"},
            {"source": "l1", "target": "t2", "type": "owned_by"},
        ]}
        self.assertFalse(has_dependency_cycle(graph))


if __name__ == "__main__":
    unittest.main()
