"""Immutable identity helpers shared by VLA data and checkpoint stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LEGACY_DATA_PROFILE = "legacy"
ROTATION_V4_DATA_PROFILE = "rotation_v4"
ROTATION_V4_INDEX_SCHEMA = "tactile_vla_v4_training_index_v1"
ROTATION_V5_DATA_PROFILE = "rotation_phase_v5"
ROTATION_V5_INDEX_SCHEMA = "tactile_vla_v5_prompt_training_index_v1"

BASE_IDENTITY_KEYS = (
    "data_profile",
    "prompt_profile",
    "data_config_hash",
    "action_frame_manifest_hash",
    "action_indices_identity",
    "index_sha256",
)
V4_IDENTITY_KEYS = (
    "selection_hash",
    "profile_config_hash",
    "training_data_hash",
    "source_file_hashes",
    "need_manifest_identity",
    "failure_manifest_identity",
    "reasoning_manifest_identity",
    "lerobot_identity",
    "norm_stats_sha256",
)
V5_IDENTITY_KEYS = (
    "experiment_kind",
    "selection_hash",
    "v4_profile_config_hash",
    "training_data_hash",
    "source_file_hashes",
    "phase_boundaries_identity",
    "action_phase_manifest_identity",
    "v4_training_data_hash",
    "v4_lerobot_identity",
    "h30_target_identity",
    "v4_norm_stats_sha256",
)
ALL_IDENTITY_KEYS = BASE_IDENTITY_KEYS + V4_IDENTITY_KEYS


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


def checkpoint_step_number(checkpoint: str | Path, *, context: str) -> int:
    """Return the positive numeric step encoded by a checkpoint directory."""

    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    if checkpoint_dir.name == "params":
        checkpoint_dir = checkpoint_dir.parent
    try:
        step = int(checkpoint_dir.name)
    except ValueError as exc:
        raise ValueError(
            f"{context} must point to a numeric checkpoint step directory; "
            f"got {checkpoint_dir}"
        ) from exc
    if step <= 0 or str(step) != checkpoint_dir.name:
        raise ValueError(
            f"{context} must point to a positive canonical checkpoint step directory; "
            f"got {checkpoint_dir}"
        )
    return step


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
    identity = {
        "data_profile": data_profile,
        "prompt_profile": prompt_profile,
        "data_config_hash": payload.get("data_config_hash"),
        "action_frame_manifest_hash": sha256_json(action_identity),
        "action_indices_identity": action_identity,
        "index_sha256": sha256_file(index_path),
        "index_file": str(Path(index_path).expanduser().resolve()),
    }

    def hashes_only(value: Any, *, version: str) -> Any:
        if isinstance(value, Mapping):
            if "sha256" in value:
                digest = str(value["sha256"])
                if len(digest) != 64:
                    raise ValueError(f"Invalid {version} source SHA256: {digest!r}")
                return digest
            return {
                str(key): hashes_only(child, version=version)
                for key, child in sorted(value.items())
            }
        digest = str(value)
        if len(digest) != 64:
            raise ValueError(f"Invalid {version} source SHA256: {digest!r}")
        return digest

    if data_profile == ROTATION_V4_DATA_PROFILE:
        if payload.get("schema_version") != ROTATION_V4_INDEX_SCHEMA:
            raise ValueError(
                "rotation_v4 requires the dedicated V4 training index schema; "
                f"got {payload.get('schema_version')!r}"
            )
        stored_training_hash = str(payload.get("training_data_hash", ""))
        calculated_training_hash = sha256_json(
            {key: value for key, value in payload.items() if key != "training_data_hash"}
        )
        if not stored_training_hash or stored_training_hash != calculated_training_hash:
            raise ValueError("V4 training_data_hash does not match the unified index payload")
        required = {
            "selection_hash": payload.get("selection_hash"),
            "profile_config_hash": payload.get("profile_config_hash"),
            "need_manifest_identity": payload.get("need_identity"),
            "failure_manifest_identity": payload.get("failure_manifest_identity"),
            "reasoning_manifest_identity": payload.get("reasoning_manifest_identity"),
            "lerobot_identity": payload.get("lerobot_identity"),
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ValueError(f"V4 unified index lacks identity fields: {missing}")
        source_files = payload.get("source_files")
        if not isinstance(source_files, Mapping) or not source_files:
            raise ValueError("V4 unified index lacks source file hashes")

        identity.update(
            {
                **required,
                "training_data_hash": stored_training_hash,
                "source_file_hashes": hashes_only(source_files, version="V4"),
            }
        )
    elif data_profile == ROTATION_V5_DATA_PROFILE:
        if payload.get("schema_version") != ROTATION_V5_INDEX_SCHEMA:
            raise ValueError(
                "rotation_phase_v5 requires the dedicated prompt-only V5 index schema; "
                f"got {payload.get('schema_version')!r}"
            )
        stored_training_hash = str(payload.get("training_data_hash", ""))
        calculated_training_hash = sha256_json(
            {key: value for key, value in payload.items() if key != "training_data_hash"}
        )
        if not stored_training_hash or stored_training_hash != calculated_training_hash:
            raise ValueError("V5 training_data_hash does not match the unified index payload")
        required = {
            "experiment_kind": payload.get("experiment_kind"),
            "selection_hash": payload.get("selection_hash"),
            "v4_profile_config_hash": payload.get("v4_profile_config_hash"),
            "phase_boundaries_identity": payload.get("phase_boundaries_identity"),
            "action_phase_manifest_identity": payload.get("action_phase_manifest_identity"),
            "v4_training_data_hash": payload.get("v4_training_data_hash"),
            "v4_lerobot_identity": payload.get("v4_lerobot_identity"),
            "h30_target_identity": payload.get("h30_target_identity"),
            "v4_norm_stats_sha256": payload.get("v4_norm_stats_sha256"),
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ValueError(f"V5 unified index lacks identity fields: {missing}")
        source_files = payload.get("source_files")
        if not isinstance(source_files, Mapping) or not source_files:
            raise ValueError("V5 unified index lacks source file hashes")
        identity.update(
            {
                **required,
                "training_data_hash": stored_training_hash,
                "source_file_hashes": hashes_only(source_files, version="V5"),
            }
        )
    return identity


def assert_identity_matches(
    saved: Mapping[str, Any],
    requested: Mapping[str, Any],
    *,
    context: str,
    keys: Sequence[str] | None = None,
) -> None:
    if keys is None:
        profiles = {saved.get("data_profile"), requested.get("data_profile")}
        if ROTATION_V4_DATA_PROFILE in profiles:
            keys = ALL_IDENTITY_KEYS
        elif ROTATION_V5_DATA_PROFILE in profiles:
            keys = BASE_IDENTITY_KEYS + V5_IDENTITY_KEYS
        else:
            keys = BASE_IDENTITY_KEYS
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
            **{key: config.get(key) for key in V4_IDENTITY_KEYS},
            **{key: config.get(key) for key in V5_IDENTITY_KEYS},
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
    if expected.get("data_profile") == ROTATION_V5_DATA_PROFILE:
        if norm_identity.get("data_profile") != ROTATION_V4_DATA_PROFILE:
            raise ValueError(f"{context} must reuse rotation_v4 norm stats")
        if norm_identity.get("action_indices_identity") != expected.get("action_indices_identity"):
            raise ValueError(f"{context} V4/V5 action indices differ")
        v4_source_hash = (
            expected.get("source_file_hashes", {}).get("v4_training_index")
            if isinstance(expected.get("source_file_hashes"), Mapping)
            else None
        )
        if v4_source_hash is not None and norm_identity.get("index_sha256") != v4_source_hash:
            raise ValueError(f"{context} was computed from a different V4 training index")
        norm_stats_path = summary_path.parent / "norm_stats.json"
        if not norm_stats_path.is_file():
            raise FileNotFoundError(norm_stats_path)
        actual_norm_hash = sha256_file(norm_stats_path)
        if (
            summary.get("norm_stats_sha256") != actual_norm_hash
            or expected.get("v4_norm_stats_sha256") != actual_norm_hash
        ):
            raise ValueError(f"{context} did not preserve the V4 norm SHA256")
        expected_train = int(expected["action_indices_identity"]["train"]["count"])
        if int(summary.get("num_frames", -1)) != expected_train:
            raise ValueError(
                f"{context} frame count mismatch: expected={expected_train}, "
                f"actual={summary.get('num_frames')}"
            )
        return summary
    norm_keys = (
        tuple(
            key
            for key in ALL_IDENTITY_KEYS
            if key not in {"prompt_profile", "norm_stats_sha256"}
        )
        if expected.get("data_profile") == ROTATION_V4_DATA_PROFILE
        else tuple(key for key in BASE_IDENTITY_KEYS if key != "prompt_profile")
    )
    assert_identity_matches(
        expected,
        norm_identity,
        context=context,
        # The norm summary records only immutable data-side identity.  Its own
        # norm_stats hash is deliberately outside artifact_identity to avoid a
        # self-reference.
        keys=norm_keys,
    )
    expected_train = int(expected["action_indices_identity"]["train"]["count"])
    actual_frames = int(summary.get("num_frames", -1))
    if actual_frames != expected_train:
        raise ValueError(
            f"{context} frame count mismatch: expected={expected_train}, actual={actual_frames}"
        )
    if expected.get("data_profile") == ROTATION_V4_DATA_PROFILE:
        stored_norm_hash = str(summary.get("norm_stats_sha256", ""))
        norm_stats_path = summary_path.parent / "norm_stats.json"
        if not norm_stats_path.is_file():
            raise FileNotFoundError(norm_stats_path)
        actual_norm_hash = sha256_file(norm_stats_path)
        if not stored_norm_hash or stored_norm_hash != actual_norm_hash:
            raise ValueError(
                f"{context} norm_stats.json hash mismatch: "
                f"stored={stored_norm_hash!r}, actual={actual_norm_hash!r}"
            )
    return summary


def selected_best_metrics_summary(
    metrics: Mapping[str, Any],
    *,
    action_loss_degradation_limit: Any,
    context: str,
) -> dict[str, Any]:
    limit = action_loss_degradation_limit
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise ValueError(f"{context} lacks action_loss_degradation_limit=0.10")
    limit = float(limit)
    if not math.isclose(limit, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context} requires action_loss_degradation_limit=0.10, got {limit!r}")
    step = metrics.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= 4_000 or step % 500:
        raise ValueError(f"{context} step must be a 500-step evaluation in [500, 4000]")
    if metrics.get("action_gate_passed") is not True:
        raise ValueError(f"{context} must have action_gate_passed=true")
    degradation = metrics.get("action_loss_degradation")
    if isinstance(degradation, bool) or not isinstance(degradation, (int, float)):
        raise ValueError(f"{context} lacks numeric action_loss_degradation")
    degradation = float(degradation)
    if not math.isfinite(degradation) or degradation > limit:
        raise ValueError(
            f"{context} exceeds action degradation limit: "
            f"degradation={degradation!r}, limit={limit!r}"
        )
    return {
        "step": step,
        "val_score": metrics.get("val_score"),
        "action_loss": metrics.get("action_loss"),
        "action_loss_baseline": metrics.get("action_loss_baseline"),
        "action_loss_degradation": degradation,
        "action_gate_passed": True,
        "action_loss_degradation_limit": limit,
        "need_macro_f1": metrics.get("need_recovery", {}).get("macro_f1"),
        "failure_exact_match": metrics.get("failure_reason", {}).get("exact_match"),
        "plan_exact_match": metrics.get("recovery_plan", {}).get("exact_match"),
    }


def validate_merged_best_metrics(
    checkpoint_root: str | Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    metrics_path = Path(checkpoint_root) / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    stored_sha = str(manifest.get("selected_best_metrics_sha256", ""))
    actual_sha = sha256_file(metrics_path)
    if len(stored_sha) != 64 or stored_sha != actual_sha:
        raise ValueError(
            f"{context} selected best metrics SHA mismatch: "
            f"stored={stored_sha!r}, actual={actual_sha!r}"
        )
    metrics = json.loads(metrics_path.read_text())
    actual_summary = selected_best_metrics_summary(
        metrics,
        action_loss_degradation_limit=config.get("action_loss_degradation_limit"),
        context=f"{context} selected best metrics",
    )
    stored_summary = manifest.get("selected_best_metrics")
    if not isinstance(stored_summary, Mapping) or dict(stored_summary) != actual_summary:
        raise ValueError(
            f"{context} selected best metrics summary mismatch: "
            f"stored={stored_summary!r}, actual={actual_summary!r}"
        )
    return actual_summary


def load_checkpoint_prompt_profile(config: Mapping[str, Any]) -> str:
    """Legacy checkpoints predate prompt profiles and must keep old prompts."""

    from tactile_vla.vla.prompts import resolve_prompt_profile

    return resolve_prompt_profile(config.get("prompt_profile"))
