from pathlib import Path
import json
import tempfile
import unittest

from deadlock_dataset.graph import GraphState, build_snapshots, read_events, write_jsonl
from deadlock_dataset.split import assign_splits, discover_runs, write_splits
from deadlock_dataset.temporal import TemporalRun, make_sequences, split_by_seed
from deadlock_dataset.validation import validate_snapshots


FIXTURE = Path(__file__).parent / "fixtures" / "abba_events.jsonl"


class GraphStateTests(unittest.TestCase):
    def test_detects_abba_cycle(self) -> None:
        state = GraphState()
        for event in read_events(FIXTURE):
            state.apply(event)
        self.assertEqual(
            state.cycle_nodes(),
            ["lock:10:0xa", "lock:10:0xb", "thread:10:11", "thread:10:12"],
        )

    def test_release_removes_cycle(self) -> None:
        state = GraphState()
        events = read_events(FIXTURE)
        for event in events:
            state.apply(event)
        state.apply({
            "run_id": "fixture-abba", "ts_ns": 110000000,
            "event": "lock_released", "pid": 10, "tid": 11, "lock_addr": "0xa",
        })
        self.assertEqual(state.cycle_nodes(), [])

    def test_mutex_allowlist_excludes_instrumentation_lock(self) -> None:
        state = GraphState(allowed_mutexes={"lock:10:0xa"})
        state.apply({
            "event": "lock_acquired", "ts_ns": 1, "pid": 10, "tid": 11,
            "lock_addr": "0xdead", "ret": 0,
        })
        self.assertEqual(state.owners, {})
        self.assertEqual(state.locks, set())

    def test_scheduler_features_are_embedded_for_known_threads(self) -> None:
        state = GraphState()
        state.apply({
            "event": "lock_acquired", "ts_ns": 1, "pid": 10, "tid": 11,
            "lock_addr": "0xa", "ret": 0,
        })
        state.apply({
            "event": "sched_switch", "ts_ns": 2, "pid": 10, "tid": 11,
            "cpu": 0, "target_tid": 12,
        })
        state.apply({
            "event": "sched_switch", "ts_ns": 3, "pid": 10, "tid": 11,
            "cpu": 1, "target_tid": 12,
        })
        features = state.snapshot("run", 3)["nodes"][0]["features"]
        self.assertEqual(features["scheduler_switches"], 2)
        self.assertEqual(features["cpu_migrations"], 1)
        self.assertEqual(features["last_cpu"], 1)


class SnapshotTests(unittest.TestCase):
    def test_labels_safe_unsafe_and_deadlocked(self) -> None:
        snapshots = build_snapshots(
            read_events(FIXTURE),
            interval_ns=10_000_000,
            unsafe_window_ns=30_000_000,
            confirm_ns=20_000_000,
        )
        labels = {snapshot["label"] for snapshot in snapshots}
        self.assertEqual(labels, {"safe", "unsafe", "deadlocked"})
        self.assertEqual(validate_snapshots(snapshots), [])

    def test_rejects_multiple_runs(self) -> None:
        events = read_events(FIXTURE)
        events.append({"run_id": "other", "ts_ns": 0, "event": "run_start"})
        with self.assertRaises(ValueError):
            build_snapshots(events, 10, 10, 10)

    def test_expected_mode_is_enforced(self) -> None:
        snapshots = build_snapshots(read_events(FIXTURE), 10_000_000, 30_000_000, 20_000_000)
        self.assertEqual(validate_snapshots(snapshots, "deadlock"), [])
        self.assertIn("safe run contains a deadlocked snapshot", validate_snapshots(snapshots, "safe"))

    def test_validator_rejects_wrong_edge_direction(self) -> None:
        snapshots = build_snapshots(read_events(FIXTURE), 10_000_000, 30_000_000, 20_000_000)
        active = next(snapshot for snapshot in snapshots if snapshot["edges"])
        active["edges"][0]["source"], active["edges"][0]["target"] = (
            active["edges"][0]["target"], active["edges"][0]["source"]
        )
        self.assertTrue(any("wrong direction" in error for error in validate_snapshots(snapshots)))

    def test_active_event_aligned_window_has_no_empty_graphs(self) -> None:
        snapshots = build_snapshots(
            read_events(FIXTURE),
            interval_ns=10_000_000,
            unsafe_window_ns=30_000_000,
            confirm_ns=20_000_000,
            start_policy="first_mutex_event",
            event_aligned=True,
            post_cycle_ns=30_000_000,
            drop_empty=True,
            deduplicate=True,
        )
        self.assertEqual(snapshots[0]["ts_ns"], 10_000_000)
        self.assertEqual(snapshots[-1]["ts_ns"], 70_000_000)
        self.assertTrue(all(snapshot["nodes"] for snapshot in snapshots))

    def test_active_allocation_window_has_edges(self) -> None:
        snapshots = build_snapshots(
            read_events(FIXTURE),
            interval_ns=10_000_000,
            unsafe_window_ns=30_000_000,
            confirm_ns=20_000_000,
            start_policy="first_graph_edge",
            event_aligned=True,
            post_cycle_ns=30_000_000,
            drop_empty=True,
            drop_edgeless=True,
            deduplicate=True,
        )
        self.assertEqual(snapshots[0]["ts_ns"], 10_000_000)
        self.assertTrue(all(snapshot["edges"] for snapshot in snapshots))


class SplitTests(unittest.TestCase):
    def test_splits_keep_runs_disjoint(self) -> None:
        snapshots = build_snapshots(read_events(FIXTURE), 10_000_000, 30_000_000, 20_000_000)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_dir = root / "runs"
            for index in range(5):
                run_id = f"run-{index}"
                run_dir = runs_dir / run_id
                copied = [{**snapshot, "run_id": run_id} for snapshot in snapshots]
                write_jsonl(run_dir / "snapshots.jsonl", copied)
                (run_dir / "run_summary.json").write_text(json.dumps({
                    "run_id": run_id, "scenario": "abba", "mode": "deadlock", "noise": "none"
                }), encoding="utf-8")
            splits = assign_splits(discover_runs(runs_dir), seed=7)
            metadata = write_splits(root / "dataset", splits, seed=7)
            run_sets = [set(metadata["splits"][name]["runs"]) for name in metadata["splits"]]
            self.assertEqual(set.union(*run_sets), {f"run-{index}" for index in range(5)})
            self.assertFalse(run_sets[0] & run_sets[1])
            self.assertFalse(run_sets[0] & run_sets[2])
            self.assertFalse(run_sets[1] & run_sets[2])

    def test_temporal_seed_splits_are_globally_disjoint(self) -> None:
        runs = []
        for seed in range(10):
            runs.append(TemporalRun(
                run_id=f"run-{seed}", run_dir=Path("."), scenario="abba",
                mode="safe", noise="none", threads=2, seed=seed,
                iterations=1, timeout_ms=1, kernel="test", machine="test",
                workload_status=0, snapshots=tuple(), sensor_counts={},
            ))
        splits = split_by_seed(runs, list(range(10)))
        seed_sets = [{run.seed for run in split} for split in splits.values()]
        self.assertEqual([len(values) for values in seed_sets], [7, 1, 2])
        self.assertFalse(seed_sets[0] & seed_sets[1])
        self.assertFalse(seed_sets[0] & seed_sets[2])
        self.assertFalse(seed_sets[1] & seed_sets[2])

    def test_temporal_sequences_reference_consecutive_snapshots(self) -> None:
        snapshots = tuple({
            "snapshot_id": f"run:{index:05d}", "ts_ns": index * 10,
            "label": "safe", "has_cycle": False,
        } for index in range(10))
        run = TemporalRun(
            run_id="run", run_dir=Path("."), scenario="abba", mode="safe",
            noise="none", threads=2, seed=1, iterations=1, timeout_ms=1,
            kernel="test", machine="test", workload_status=0,
            snapshots=snapshots, sensor_counts={},
        )
        sequences = make_sequences(run, length=4, stride=2)
        self.assertEqual(len(sequences), 4)
        self.assertEqual(sequences[0]["snapshot_ids"], [
            "run:00000", "run:00001", "run:00002", "run:00003"
        ])


if __name__ == "__main__":
    unittest.main()
