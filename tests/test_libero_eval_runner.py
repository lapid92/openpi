import json

import pytest

from scripts import libero_eval_runner


def _manifest(tmp_path, **overrides):
    payload = {
        "artifact_id": "volt-p0z8paae9oa0-no-tvm-29999",
        "artifact_name": "pi05-libero-base-seed42-30k",
        "model_class": "finetune",
        "training_run_id": "pi05_libero_base_seed42_30k_20260914",
        "checkpoint_label": "29999",
        "checkpoint_name": "29999",
        "checkpoint_path": "/volt/data/openpi_evals/checkpoints/pi05_libero_base_seed42_30k_20260914/29999",
        "checkpoint_s3_uri": "",
        "checkpoint_volt_job_id": "p0z8paae9oa0",
        "checkpoint_volt_artifact_prefix": "openpi_checkpoints/pi05_libero_base_seed42_30k_20260914/29999",
        "train_steps_completed": 30000,
        "config": "pi05_libero",
        "git_sha": "d3b0cbe119cb11abc51561071751d7823b9222e4",
    }
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accepts_direct_volt_artifact_provenance(tmp_path):
    artifact = libero_eval_runner.Artifact.from_json(_manifest(tmp_path))

    assert artifact.checkpoint_volt_job_id == "p0z8paae9oa0"
    assert artifact.checkpoint_s3_uri == ""


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"checkpoint_volt_job_id": ""}, "job ID"),
        ({"checkpoint_volt_job_id": "", "checkpoint_volt_artifact_prefix": ""}, "exactly one"),
        ({"checkpoint_volt_artifact_prefix": "../checkpoint"}, "unsafe"),
        ({"artifact_id": "sha256:" + "0" * 64}, "must start"),
        ({"checkpoint_s3_uri": "s3://unexpected"}, "exactly one"),
    ],
)
def test_rejects_invalid_volt_artifact_provenance(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        libero_eval_runner.Artifact.from_json(_manifest(tmp_path, **overrides))
