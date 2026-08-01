"""Scoring, integrity checks, and batch aggregation."""

from __future__ import annotations

import json

from athanor.cc_harness.scoring import aggregate, contamination_scan, format_aggregate, score_final

EXPECTED = [[[3, 0, 6], [0, 4, 0]]]


def _final(candidates):
    return {"test": [{"index": 0, "candidates": candidates}]}


class TestScoring:
    def test_exact_first_candidate_solves(self):
        score = score_final(_final([[[3, 0, 6], [0, 4, 0]]]), EXPECTED)
        assert score["solved"] is True
        assert score["score"] == 1.0
        assert score["per_test"][0]["matched_candidate"] == 1

    def test_second_candidate_also_counts(self):
        score = score_final(_final([[[0, 0, 0]], [[3, 0, 6], [0, 4, 0]]]), EXPECTED)
        assert score["solved"] is True
        assert score["per_test"][0]["matched_candidate"] == 2

    def test_wrong_prediction_scores_zero(self):
        score = score_final(_final([[[1, 1, 1]]]), EXPECTED)
        assert score["solved"] is False
        assert score["score"] == 0.0

    def test_partial_credit_across_test_examples(self):
        final = {
            "test": [
                {"index": 0, "candidates": [[[1]]]},
                {"index": 1, "candidates": [[[2]]]},
            ]
        }
        score = score_final(final, [[[9]], [[2]]])
        assert score["num_solved"] == 1
        assert score["score"] == 0.5
        assert score["solved"] is False

    def test_unaccepted_run_is_unscored(self):
        score = score_final(None, EXPECTED)
        assert score["scored"] is False
        assert score["solved"] is False


class TestContamination:
    def test_clean_run_is_clean(self, workspace, tmp_path):
        stream = tmp_path / "stream.jsonl"
        stream.write_text('{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n')
        report = contamination_scan(
            workspace_root=workspace.root, stream_path=stream, dataset_root=tmp_path / "dataset",
            task_id="mirror01",
        )
        assert report["suspected"] is False

    def test_dataset_root_reference_is_flagged(self, workspace, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        stream = tmp_path / "stream.jsonl"
        stream.write_text(json.dumps({"cmd": f"cat {dataset.resolve()}/mirror01.json"}) + "\n")
        report = contamination_scan(
            workspace_root=workspace.root, stream_path=stream, dataset_root=dataset, task_id="mirror01"
        )
        assert report["suspected"] is True
        assert any("dataset root" in item for item in report["evidence"])

    def test_out_of_workspace_task_file_is_flagged(self, workspace, tmp_path):
        stream = tmp_path / "stream.jsonl"
        stream.write_text('{"cmd":"cat /data/ARC-AGI-2/evaluation/mirror01.json"}\n')
        report = contamination_scan(
            workspace_root=workspace.root, stream_path=stream, task_id="mirror01"
        )
        assert report["suspected"] is True

    def test_workspace_task_file_reference_is_not_flagged(self, workspace, tmp_path):
        stream = tmp_path / "stream.jsonl"
        stream.write_text('{"cmd":"cat task/task.json"}\n')
        report = contamination_scan(
            workspace_root=workspace.root, stream_path=stream, task_id="mirror01"
        )
        assert report["suspected"] is False

    def test_network_tool_use_is_flagged(self, workspace, tmp_path):
        stream = tmp_path / "stream.jsonl"
        stream.write_text('{"type":"assistant","message":{"content":[{"type":"tool_use","name":"WebSearch"}]}}\n')
        report = contamination_scan(workspace_root=workspace.root, stream_path=stream)
        assert report["suspected"] is True

    def test_tampered_task_file_is_flagged(self, workspace):
        path = workspace.root / "task" / "task.json"
        task = json.loads(path.read_text(encoding="utf-8"))
        task["test"][0]["output"] = [[1]]
        path.write_text(json.dumps(task), encoding="utf-8")
        report = contamination_scan(workspace_root=workspace.root)
        assert report["suspected"] is True
        assert any("tampered" in item for item in report["evidence"])


class TestAggregate:
    def _records(self):
        return [
            {
                "task_id": "a",
                "accepted": True,
                "cost_usd": 2.0,
                "iterations_used": 3,
                "score": {"scored": True, "solved": True, "score": 1.0, "num_solved": 1, "num_test": 1},
            },
            {
                "task_id": "b",
                "accepted": True,
                "cost_usd": 4.0,
                "iterations_used": 7,
                "score": {"scored": True, "solved": False, "score": 0.5, "num_solved": 1, "num_test": 2},
            },
            {"task_id": "c", "error": "wall-clock timeout", "iterations_used": 12,
             "score": {"scored": False, "solved": False, "score": 0.0, "num_solved": 0, "num_test": 1}},
        ]

    def test_counts_and_costs(self):
        summary = aggregate(self._records())
        assert summary["tasks"] == 3
        assert summary["scored"] == 2
        assert summary["solved"] == 1
        assert summary["accuracy"] == 0.5
        assert summary["partial_credit"] == 0.75
        assert summary["mean_cost_usd"] == 3.0
        assert summary["errors"] == ["c"]

    def test_formats_a_readable_table(self):
        records = self._records()
        text = format_aggregate(aggregate(records), records)
        assert "solved     : 1/2" in text
        assert "wall-clock timeout" in text
