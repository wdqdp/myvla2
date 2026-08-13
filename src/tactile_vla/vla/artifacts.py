"""Immutable identity helpers shared by VLA data and checkpoint stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


LEGACY_DATA_PROFILE = "legacy"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_indices_identity(splits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    all_indices: list[int] = []
    for split in ("train", "val", "test"):
        indices = [int(value) for value in splits[split]["execution_indices"]]
        identity[split] = {
            "count": len(indices),
            "sha256": sha256_json(indices),
        }
        all_indices.extend(indices)
    identity["all"] = {
        "count": len(all_indices),
        "sha256": sha256_json(all_indices),
    }
    return identity


def validate_action_indices_identity(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    actual = action_indices_identity(payload["splits"])
    stored = payload.get("action_indices_identity")
    if stored is not None and stored != actual:
        raise ValueError(
            "Index action frame manifest does not match its execution_indices: "
            f"stored={stored}, actual={actual}"
        )
    if expected is not None and dict(expected) != actual:
        raise ValueError(
            "Action frame manifest mismatch: "
            f"expected={dict(expected)}, actual={actual}"
        )
    return actual


def artifact_identity(
    payload: Mapping[str, Any],
    *,
    index_path: str | Path,
    prompt_profile: str,
    requested_data_profile: str,
) -> dict[str, Any]:
    data_profile = str(payload.get("data_profile", LEGACY_DATA_PROFILE))
    if data_profile != requested_data_profile:
        raise ValueError(
            f"Index data_profile={data_profile!r}, requested {requested_data_profile!r}"
        )
    action_identity = validate_action_indices_identity(payload)
    return {
        "data_profile": data_profile,
        "prompt_profile": prompt_profile,
        "data_config_hash": payload.get("data_config_hash"),
        "action_frame_manifest_hash": sha256_json(action_identity),
        "action_indices_identity": action_identity,
        "index_sha256": sha256_file(index_path),
        "index_file": str(Path(index_path).expanduser().resolve()),
    }


def assert_identity_matches(
    saved: Mapping[str, Any],
    requested: Mapping[str, Any],
    *,
    context: str,
    keys: Sequence[str] = (
        "data_profile",
        "prompt_profile",
        "data_config_hash",
        "action_frame_manifest_hash",
        "action_indices_identity",
        "index_sha256",
    ),
) -> None:
    for key in keys:
        if saved.get(key) != requested.get(key):
            raise ValueError(
                f"{context} artifact identity mismatch for {key}: "
                f"saved={saved.get(key)!r}, requested={requested.get(key)!r}"
            )


def checkpoint_artifact_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    identity = config.get("artifact_identity")
    if not isinstance(identity, Mapping):
        # Checkpoints created before profiles intentionally retain legacy
        # behavior, but cannot be resumed as a versioned-profile run.
        return {
            "data_profile": str(config.get("data_profile", LEGACY_DATA_PROFILE)),
            "prompt_profile": str(config.get("prompt_profile", "legacy")),
            "data_config_hash": config.get("data_config_hash"),
            "action_frame_manifest_hash": config.get("action_frame_manifest_hash"),
            "action_indices_identity": config.get("action_indices_identity"),
            "index_sha256": config.get("index_sha256"),
            "index_file": config.get("index_file"),
        }
    return dict(identity)


def validate_norm_stats_identity(
    summary_path: str | Path,
    expected: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    norm_identity = summary.get("artifact_identity", {})
    assert_identity_matches(
        expected,
        norm_identity,
        context=context,
        keys=(
            "data_profile",
            "data_config_hash",
            "action_frame_manifest_hash",
            "action_indices_identity",
            "index_sha256",
        ),
    )
    expected_train = int(expected["action_indices_identity"]["train"]["count"])
    actual_frames = int(summary.get("num_frames", -1))
    if actual_frames != expected_train:
        raise ValueError(
            f"{context} frame count mismatch: expected={expected_train}, actual={actual_frames}"
        )
    return summary


def load_checkpoint_prompt_profile(config: Mapping[str, Any]) -> str:
    """Legacy checkpoints predate prompt profiles and must keep old prompts."""

    from tactile_vla.vla.prompts import resolve_prompt_profile

    return resolve_prompt_profile(config.get("prompt_profile"))
