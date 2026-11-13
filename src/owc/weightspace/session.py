"""Content-verified, immutable revisions for auditable weight-edit sessions."""

import json
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from owc.board.encoding import ENCODING_VERSION, encode_board, parse_legal_fen
from owc.models.raw_mlp import RawMLP
from owc.weightspace.patches import apply_patch, inspect_weights, model_hash

REVISION_SCHEMA = "owc.weightrevision.v1"
LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
REVISION_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _file_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_label(name: str) -> None:
    if not isinstance(name, str) or LABEL_PATTERN.fullmatch(name) is None:
        raise ValueError("Checkpoint name must be 1-64 alphanumeric, '-' or '_' characters")


@contextmanager
def _lock(root: Path):
    """Fail closed on concurrent writes; remove stale locks manually after a crash."""
    lock = root / ".write.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Weight session is locked: {lock}") from error
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


class WeightSession:
    """Manage a checkpoint DAG rooted at an immutable, zero-weight revision."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        if not (self.directory / "HEAD").is_file():
            raise FileNotFoundError(f"Weight session not initialized: {directory}")

    @classmethod
    def create(cls, directory: Path) -> "WeightSession":
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=False)
        try:
            (root / "revisions").mkdir()
            (root / "checkpoints").mkdir()
            obj = object.__new__(cls)
            obj.directory = root
            obj._commit(
                RawMLP().eval(),
                parent=None,
                action="init",
                data={"note": "Zero-initialized OWC RawMLP"},
            )
            return cls(root)
        except BaseException:
            shutil.rmtree(root)
            raise

    def head(self) -> str:
        revision = (self.directory / "HEAD").read_text().strip()
        if REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError("Invalid revision HEAD")
        return revision

    def _load(self, revision: str) -> tuple[RawMLP, dict]:
        if REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError("Invalid revision identifier")
        base = self.directory / "revisions" / revision
        meta = json.loads((base / "manifest.json").read_text())
        if meta["schema_version"] != REVISION_SCHEMA or meta["revision"] != revision:
            raise ValueError("Invalid revision manifest")
        if meta["encoding_version"] != ENCODING_VERSION:
            raise ValueError("Encoding mismatch; cannot load incompatible revision")
        weights_file = base / "weights.npz"
        if meta["weights_sha256"] != _file_hash(weights_file):
            raise ValueError("Checkpoint byte hash mismatch")

        model = RawMLP().eval()
        parameters = dict(model.named_parameters())
        with np.load(weights_file, allow_pickle=False) as stored:
            if set(stored.files) != set(parameters):
                raise ValueError("Checkpoint tensor names do not match model")
            with torch.no_grad():
                for name, param in parameters.items():
                    array = stored[name]
                    if array.dtype != np.dtype("float32") or array.shape != tuple(param.shape):
                        raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
                    if not bool(np.isfinite(array).all()):
                        raise ValueError(f"Checkpoint has non-finite tensor: {name}")
                    param.copy_(torch.from_numpy(array.copy()))
        if model_hash(model) != meta["model_hash"]:
            raise ValueError("Checkpoint model hash mismatch")
        return model, meta

    def current(self) -> tuple[RawMLP, dict]:
        return self._load(self.head())

    def _commit(self, model: RawMLP, *, parent: str | None, action: str, data: dict) -> dict:
        rev = uuid.uuid4().hex
        root = self.directory
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root / "revisions"))
        published = root / "revisions" / rev
        try:
            file = temporary / "weights.npz"
            arrays = {
                name: param.detach().cpu().numpy().copy()
                for name, param in model.named_parameters()
            }
            np.savez_compressed(file, **arrays)
            meta = {
                "schema_version": REVISION_SCHEMA,
                "encoding_version": ENCODING_VERSION,
                "revision": rev,
                "parent": parent,
                "action": action,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "model_hash": model_hash(model),
                "weights_sha256": _file_hash(file),
                "data": data,
            }
            _write_json(temporary / "manifest.json", meta)
            os.rename(temporary, published)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, prefix=".HEAD-", delete=False
            ) as temp_head:
                temp_name = Path(temp_head.name)
                temp_head.write(rev + "\n")
            try:
                os.replace(temp_name, root / "HEAD")
            finally:
                temp_name.unlink(missing_ok=True)
            return meta
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def inspect(self, *, tensor: str | None = None, max_values: int = 12) -> dict:
        model, meta = self.current()
        return {
            "revision": meta["revision"],
            "parent": meta["parent"],
            "action": meta["action"],
            **inspect_weights(model, tensor=tensor, max_values=max_values),
        }

    def probe(self, fen: str) -> dict:
        model, meta = self.current()
        board = parse_legal_fen(fen)
        with torch.inference_mode():
            result = float(model(encode_board(board)).item())
        return {
            "fen": board.fen(en_passant="legal"),
            "revision": meta["revision"],
            "model_hash": meta["model_hash"],
            "score": result,
            "score_perspective": "side_to_move",
        }

    def apply(self, patch: dict) -> dict:
        with _lock(self.directory):
            old = self.head()
            model, meta = self._load(old)
            result = apply_patch(model, patch)
            if result["before_hash"] != meta["model_hash"]:
                raise ValueError("Invalid expected base hash")
            updated = self._commit(
                model,
                parent=old,
                action="patch",
                data={"patch": patch, "result": result},
            )
            return {"revision": updated["revision"], **result}

    def checkpoint(self, name: str) -> dict:
        _validate_label(name)
        with _lock(self.directory):
            current = self.head()
            _, meta = self._load(current)
            target = self.directory / "checkpoints" / f"{name}.json"
            marker = {
                "name": name,
                "revision": current,
                "model_hash": meta["model_hash"],
            }
            with target.open("x", encoding="utf-8") as stream:
                json.dump(marker, stream, sort_keys=True)
                stream.write("\n")
            return marker

    def rollback(self, name: str) -> dict:
        _validate_label(name)
        with _lock(self.directory):
            current = self.head()
            before_model, _ = self._load(current)
            marker = json.loads(
                (self.directory / "checkpoints" / f"{name}.json").read_text()
            )
            if marker["name"] != name:
                raise ValueError("Checkpoint marker name mismatch")
            restored, meta = self._load(marker["revision"])
            if marker["model_hash"] != meta["model_hash"]:
                raise ValueError("Checkpoint marker hash mismatch")
            updated = self._commit(
                restored,
                parent=current,
                action="rollback",
                data={
                    "checkpoint": name,
                    "restored_from": marker["revision"],
                    "before_hash": model_hash(before_model),
                    "after_hash": model_hash(restored),
                },
            )
            return {
                "revision": updated["revision"],
                "before_hash": model_hash(before_model),
                "after_hash": updated["model_hash"],
                "restored_from": marker["revision"],
                "checkpoint": name,
            }
