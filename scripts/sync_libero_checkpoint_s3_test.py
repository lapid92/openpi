import io
import json
import pathlib
import tarfile

import pytest

from . import sync_libero_checkpoint_s3 as checkpoint_sync


class NotFoundError(Exception):
    def __init__(self):
        self.response = {"ResponseMetadata": {"HTTPStatusCode": 404}, "Error": {"Code": "NoSuchKey"}}


class PreconditionFailedError(Exception):
    def __init__(self):
        self.response = {"ResponseMetadata": {"HTTPStatusCode": 412}, "Error": {"Code": "PreconditionFailed"}}


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.operations: list[tuple[str, str]] = []

    def head_object(self, **kwargs):
        bucket = kwargs["Bucket"]
        key = kwargs["Key"]
        try:
            value = self.objects[(bucket, key)]
        except KeyError as exc:
            raise NotFoundError() from exc
        return {"ContentLength": len(value)}

    def upload_file(self, filename, bucket, key):
        self.operations.append(("upload_file", key))
        self.objects[(bucket, key)] = pathlib.Path(filename).read_bytes()

    def put_object(self, **kwargs):
        object_key = (kwargs["Bucket"], kwargs["Key"])
        if kwargs.get("IfNoneMatch") == "*" and object_key in self.objects:
            raise PreconditionFailedError
        self.operations.append(("put_object", kwargs["Key"]))
        self.objects[object_key] = kwargs["Body"]

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(self.objects[(kwargs["Bucket"], kwargs["Key"])])}

    def download_file(self, bucket, key, filename):
        pathlib.Path(filename).write_bytes(self.objects[(bucket, key)])


@pytest.fixture(autouse=True)
def volt_environment(monkeypatch):
    monkeypatch.setenv("VOLT_CLUSTER_NAME", "test-cluster")
    monkeypatch.setattr(checkpoint_sync.pathlib.Path, "exists", _exists_with_volt(checkpoint_sync.pathlib.Path.exists))


def _exists_with_volt(original_exists):
    def exists(path):
        if str(path) == "/volt":
            return True
        return original_exists(path)

    return exists


def _make_checkpoint(root: pathlib.Path) -> pathlib.Path:
    checkpoint = root / "29999"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets" / "physical-intelligence" / "libero").mkdir(parents=True)
    (checkpoint / "params" / "weights").write_bytes(b"weights")
    (checkpoint / "assets" / "physical-intelligence" / "libero" / "norm_stats.json").write_text("{}")
    (checkpoint / "_CHECKPOINT_METADATA").write_text('{"step": 30000}')
    (checkpoint / "train_state").mkdir()
    (checkpoint / "train_state" / "state").write_bytes(b"must not upload")
    return checkpoint


def _metadata(model_class: str = "tvm") -> dict:
    return {
        "artifact_name": "pi05-libero-tvm-30k",
        "model_class": model_class,
        "training_run_id": "pi05_libero_tvm_matched_b256_seed42_30k_20260918",
        "checkpoint_name": "29999",
        "checkpoint_label": "29999",
        "train_steps_completed": 30000,
        "config": "pi05_libero_tvm",
        "git_sha": "a" * 40,
    }


ARTIFACT_KEY = "tvm/pi05_libero_tvm_matched_b256_seed42_30k_20260918/step_30000-checkpoint_29999"


def test_upload_then_download_round_trip_excludes_train_state(tmp_path):
    source = _make_checkpoint(tmp_path / "source")
    s3 = FakeS3()

    manifest = checkpoint_sync.upload(source, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=s3)
    published_key = manifest["artifact_key"]

    assert manifest["bucket"] == "aair-users-east-2"
    assert manifest["prefix"] == f"arilap01/libero/openpi/checkpoints/{published_key}/"
    assert manifest["region"] == "us-east-2"
    assert [operation for operation, _ in s3.operations] == ["upload_file", "put_object", "put_object"]
    assert [key.rsplit("/", 1)[-1] for _, key in s3.operations] == [
        checkpoint_sync.ARCHIVE_NAME,
        checkpoint_sync.MANIFEST_NAME,
        checkpoint_sync.SUCCESS_NAME,
    ]

    destination = tmp_path / "restored"
    checkpoint_sync.download(destination, published_key, s3=s3)
    assert (destination / "params" / "weights").read_bytes() == b"weights"
    assert (destination / "assets" / "physical-intelligence" / "libero" / "norm_stats.json").is_file()
    assert (destination / "_CHECKPOINT_METADATA").is_file()
    assert not (destination / "train_state").exists()


def test_upload_refuses_existing_remote_object(tmp_path):
    source = _make_checkpoint(tmp_path)
    s3 = FakeS3()
    checkpoint_sync.upload(source, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=s3)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        checkpoint_sync.upload(source, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=s3)


def test_intermediate_zero_based_checkpoint_label_is_allowed():
    metadata = _metadata()
    metadata["checkpoint_name"] = "4999"
    metadata["checkpoint_label"] = "4999"
    metadata["train_steps_completed"] = 5000

    checkpoint_sync._validate_metadata(  # noqa: SLF001
        "tvm/pi05_libero_tvm_matched_b256_seed42_30k_20260918/step_5000-checkpoint_4999",
        metadata,
    )


def test_metadata_rejects_config_for_wrong_model_class():
    metadata = _metadata()
    metadata["config"] = "pi05_libero"

    with pytest.raises(ValueError, match="config must be pi05_libero_tvm"):
        checkpoint_sync._validate_metadata(ARTIFACT_KEY, metadata)  # noqa: SLF001


def test_upload_rejects_checkpoint_metadata_step_mismatch(tmp_path):
    source = _make_checkpoint(tmp_path)
    (source / "_CHECKPOINT_METADATA").write_text('{"step": 123}')

    with pytest.raises(ValueError, match="does not match train_steps_completed"):
        checkpoint_sync.upload(source, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=FakeS3())


def test_conditional_publication_refuses_manifest_overwrite():
    s3 = FakeS3()
    key = checkpoint_sync._key(ARTIFACT_KEY, checkpoint_sync.MANIFEST_NAME)  # noqa: SLF001
    s3.objects[(checkpoint_sync.S3_BUCKET, key)] = b"existing"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        checkpoint_sync._put_object_if_absent(s3, key, b"replacement")  # noqa: SLF001
    assert s3.objects[(checkpoint_sync.S3_BUCKET, key)] == b"existing"


def test_download_requires_success_marker(tmp_path):
    s3 = FakeS3()
    with pytest.raises(FileNotFoundError, match="missing _SUCCESS"):
        checkpoint_sync.download(tmp_path / "destination", ARTIFACT_KEY, s3=s3)


def test_download_rejects_checksum_mismatch_without_destination(tmp_path):
    source = _make_checkpoint(tmp_path / "source")
    s3 = FakeS3()
    uploaded = checkpoint_sync.upload(source, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=s3)
    published_key = uploaded["artifact_key"]
    manifest_key = checkpoint_sync._key(published_key, checkpoint_sync.MANIFEST_NAME)  # noqa: SLF001
    manifest = json.loads(s3.objects[(checkpoint_sync.S3_BUCKET, manifest_key)])
    manifest["archive_sha256"] = "0" * 64
    s3.objects[(checkpoint_sync.S3_BUCKET, manifest_key)] = json.dumps(manifest).encode()
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="SHA256"):
        checkpoint_sync.download(destination, published_key, s3=s3)
    assert not destination.exists()


def test_download_rejects_remote_size_mismatch_without_destination(tmp_path):
    source = _make_checkpoint(tmp_path / "source")
    s3 = FakeS3()
    uploaded = checkpoint_sync.upload(source, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=s3)
    published_key = uploaded["artifact_key"]
    archive_key = checkpoint_sync._key(published_key, checkpoint_sync.ARCHIVE_NAME)  # noqa: SLF001
    s3.objects[(checkpoint_sync.S3_BUCKET, archive_key)] = b"truncated"
    destination = tmp_path / "destination"

    with pytest.raises(ValueError, match="Remote archive size"):
        checkpoint_sync.download(destination, published_key, s3=s3)
    assert not destination.exists()


def test_download_refuses_existing_destination(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite destination"):
        checkpoint_sync.download(destination, ARTIFACT_KEY, s3=FakeS3())


def test_archive_validation_rejects_traversal():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    stream.seek(0)
    with (
        tarfile.open(fileobj=stream, mode="r") as archive,
        pytest.raises(ValueError, match="Unsafe archive member"),
    ):
        checkpoint_sync._validate_archive_members(archive)  # noqa: SLF001


@pytest.mark.parametrize("artifact_key", ["../escape", "nested/path", "", ".hidden", "space id"])
def test_artifact_key_must_match_approved_hierarchy(artifact_key):
    with pytest.raises(ValueError, match="artifact_key"):
        checkpoint_sync._key(artifact_key, checkpoint_sync.ARCHIVE_NAME)  # noqa: SLF001


def test_requires_volt_path_and_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("VOLT_CLUSTER_NAME")
    with pytest.raises(RuntimeError, match="restricted to Volt"):
        checkpoint_sync.upload(tmp_path, ARTIFACT_KEY, _metadata(), scratch_dir=tmp_path, s3=FakeS3())


def test_s3_client_uses_fixed_region(monkeypatch):
    calls = []

    class FakeBoto3:
        @staticmethod
        def client(service, **kwargs):
            calls.append((service, kwargs))
            return object()

    monkeypatch.setattr(checkpoint_sync.importlib, "import_module", lambda name: FakeBoto3)

    checkpoint_sync._get_s3_client()  # noqa: SLF001

    assert calls == [("s3", {"region_name": "us-east-2"})]
