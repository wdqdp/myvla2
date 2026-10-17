"""Joint blackbox-new-environment / book V9.5 data, without legacy weights."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re

import numpy as np

from tactile_vla.vla.artifacts import sha256_file, sha256_json
from tactile_vla.vla.v5_3_phase_change import StateQuantileStats, discretize_state_qpos
from tactile_vla.vla.v7_7_multitask_data import (
    NEGATIVE_SOURCES,
    allocate_balanced_without_replacement,
    deterministic_uniform_select,
    load_jsonl,
    stable_priority,
)
from tactile_vla.vla.v7_7_phase_prompt import compact_history

ROOT = Path("/data1/qxh/tac_vla_new/tac_data/demon_data")
OUTPUT = ROOT / "joint_blackbox_book/outputs/v12_1_data"
DEFAULT_INDEX = OUTPUT / "v12_1_training_index.json"
DATA_PROFILE = "v12_1_blackbox_new_env_book_v9_5"
DOMAINS = ("blackbox", "book")
SPLITS = ("train", "val", "test")
TASKS = ("action", "adjustment", "need", "failure", "plan")
DIRECTION_MAP = {
    "blackbox": {"left": "counterclockwise", "right": "clockwise"},
    "book": {"left": "clockwise", "right": "counterclockwise"},
}
SOURCES = {
    "blackbox": ROOT / "black_box/outputs/rotation_v7_7_2_multitask/v7_7_2_multitask_training_index.json",
    "book": ROOT / "book/outputs/book_v9_5_multitask/book_v9_5_multitask_training_index.json",
}
DATASETS = {
    "blackbox": ROOT / "black_box/lerobot_data/tactile_vla_rotation_v4_1_new_env",
    "book": ROOT / "book/lerobot_data/tactile_vla_rotation_v4",
}
POLICY = {
    "blackbox_scope": "original_episode_261_350_only_no_old_environment",
    "book_scope": "existing_book_v9_5_selection_exclude_history9_4_5",
    "blackbox_need": "original_positive_keep_correct_rotation_only_exclude_none_and_opposite_never_relabel",
    "blackbox_reasoning": "original_failure_plan_rows_keep_correct_current_rotation_only",
    "need_negative_ratio": "1_positive:3_negative_per_domain_per_split",
    "blackbox_hard_negatives": "reserve_preF30_then_uniform_other_preF",
    "action": "unchanged_source_phase_pure_raw_contiguous_H30",
    "history": "retain_source_episode_continuous_H100_11_source_points_rediscretize_shared_norm",
    "normalization": "exact_equal_domain_mass_train_action_origins_state_and_H30_delta_actions",
    "touch_missing_fields": "Fz_bias=unknown_when_not_annotated_not_none",
    "direction_mapping": DIRECTION_MAP,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def seal(payload):
    payload["training_data_hash"] = sha256_json({k: v for k, v in payload.items() if k != "training_data_hash"})
    return payload


def rotation(prompt):
    matches = re.findall(r"(?<![A-Za-z_])rotation=(none|clockwise|counterclockwise)(?=;|\])", prompt)
    if len(matches) != 1:
        raise ValueError("Expected exactly one current tactile rotation")
    return matches[0]


def canonical_touch(prompt):
    """Keep measurements and directions; an absent bias is unknown, not neutral."""

    def replace(match):
        text = match.group(0)
        if not re.search(r"(?:\[|;)\s*Fz_bias=", text):
            text = re.sub(r";\s*rotation=", "; Fz_bias=unknown; rotation=", text)
        return text

    return re.sub(r"Touch\[[^\]]+\]", replace, prompt)


def correct_rotation(row, domain, metas):
    key = (int(row["episode_id"]), int(row["attempt_id"]))
    direction = metas[key]["rotation_direction"]
    if direction not in DIRECTION_MAP[domain]:
        raise ValueError(f"Failure observation has no left/right direction: {domain}/{key}")
    return rotation(row["prompt"]) == DIRECTION_MAP[domain][direction]


def weighted_statistics(parts):
    """Exact mixture quantiles: each physical task has weight 1/2, regardless of size."""
    if len(parts) != 2 or any(len(p) == 0 for p in parts):
        raise ValueError("Both domains must contribute nonempty train-only statistics")
    arrays = [np.asarray(p, dtype=np.float64).reshape(-1, 7) for p in parts]
    if any(not np.isfinite(p).all() for p in arrays):
        raise ValueError("Nonfinite normalization input")
    mean = np.mean([p.mean(axis=0) for p in arrays], axis=0)
    second = np.mean([(p * p).mean(axis=0) for p in arrays], axis=0)
    values = np.concatenate(arrays)
    weights = np.concatenate([np.full(len(p), 0.5 / len(p)) for p in arrays])
    qs = []
    for dim in range(7):
        order = np.argsort(values[:, dim], kind="stable")
        cumulative = np.cumsum(weights[order])
        qs.append([values[order[min(np.searchsorted(cumulative, q), len(order) - 1)], dim] for q in (0.01, 0.99)])
    qs = np.asarray(qs)
    if np.any(qs[:, 1] <= qs[:, 0]):
        raise ValueError("Degenerate shared quantile bounds")
    return {
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(second - mean * mean, 0)).tolist(),
        "q01": qs[:, 0].tolist(),
        "q99": qs[:, 1].tolist(),
    }


def load_columns(dataset):
    import pyarrow.parquet as pq

    columns = [
        "index",
        "episode_id",
        "attempt_id",
        "frame_index",
        "ros_timestamp",
        "observation.state",
        "action",
        "instruction",
        "tactile_caption",
        "input_recovery_plan",
    ]
    result = {}
    for path in sorted((Path(dataset) / "data").glob("chunk-*/episode_*.parquet")):
        data = pq.ParquetFile(path).read(columns=columns).to_pydict()
        for i, global_index in enumerate(data["index"]):
            key = int(global_index)
            if key in result:
                raise ValueError("Duplicate LeRobot global index")
            result[key] = {c: data[c][i] for c in columns}
    if not result:
        raise ValueError(f"Empty dataset: {dataset}")
    return result


def _register(hashes, path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    hashes[str(path)] = digest


def load_sources(paths):
    """Read data-only source closure. Never validate/load a historical Stage A model."""
    contexts, hashes = {}, {}
    for domain in DOMAINS:
        path = Path(paths[domain]).resolve()
        _register(hashes, path)
        index = read_json(path)
        if index["training_data_hash"] != sha256_json({k: v for k, v in index.items() if k != "training_data_hash"}):
            raise ValueError(f"Bad source index hash: {domain}")
        dataset = Path(index["dataset_dir"]).resolve()
        if dataset != DATASETS[domain].resolve():
            raise ValueError(f"Unexpected {domain} dataset; old blackbox data are forbidden")
        if domain == "book" and index.get("experiment_version") != "book_v9_5":
            raise ValueError("Book must use the V9.5 source index")
        # Source indexes also record old models, but they are not dependencies of data construction.
        for name, digest in index.get("source_hashes", {}).items():
            if "stage_a_action" not in Path(name).parts:
                if "history9_4_5" in Path(name).resolve().parts:
                    raise ValueError("Archived book data are forbidden")
                _register(hashes, name, digest)
        action_path = Path(index["action_index_file"])
        action = read_json(action_path)
        _register(hashes, action_path)
        if action["training_data_hash"] != index["action_training_data_hash"]:
            raise ValueError("Source action identity differs from multitask index")
        for source in action["source_files"].values():
            _register(hashes, source["path"], source["sha256"])
        phase_path = Path(action["source_files"]["action_phase_manifest"]["path"])
        lookup = {int(r["global_index"]): r for r in load_jsonl(phase_path)}
        v4_path = Path(action["source_files"]["v4_training_index"]["path"])
        profile_path = v4_path.parent / "profile.json"
        _register(hashes, profile_path)
        metas = {(int(r["episode_id"]), int(r["attempt_id"])): r for r in read_json(profile_path)["attempts"]}
        frames = load_columns(dataset)
        episode_ids = {int(r["episode_id"]) for r in frames.values()}
        if domain == "blackbox" and episode_ids != set(range(261, 351)):
            raise ValueError("Blackbox must contain exactly original episode 261..350")
        if set((int(r["episode_id"]), int(r["attempt_id"])) for r in frames.values()) != set(metas):
            raise ValueError("Source profile and LeRobot attempts differ")
        for p in sorted((dataset / "data").glob("chunk-*/episode_*.parquet")):
            _register(hashes, p)
        for p in sorted((dataset / "meta").glob("*")):
            if p.is_file():
                _register(hashes, p)
        manifests = {}
        for task in TASKS[1:]:
            p = Path(index[f"{task}_manifest_file"])
            _register(hashes, p, index[f"{task}_manifest_sha256"])
            manifests[task] = load_jsonl(p)
        contexts[domain] = {
            "index": index,
            "dataset": dataset,
            "action": action,
            "lookup": lookup,
            "metas": metas,
            "frames": frames,
            "manifests": manifests,
        }
    return contexts, hashes


def action_delta_chunks(ctx, indices):
    states, chunks = [], []
    for g in indices:
        g = int(g)
        anchor, phase = ctx["frames"][g], ctx["lookup"][g]
        if not phase["trainable"] or not phase["chunk_phase_pure"]:
            raise ValueError("Source action must be trainable and phase pure")
        state = np.asarray(anchor["observation.state"], dtype=np.float32)
        values = []
        for offset in range(30):
            r = ctx["frames"][g + offset]
            if (r["episode_id"], r["attempt_id"], r["frame_index"]) != (
                anchor["episode_id"],
                anchor["attempt_id"],
                anchor["frame_index"] + offset,
            ):
                raise ValueError("H30 crosses an attempt")
            if phase["phase"] == "adjustment" and ctx["lookup"][g + offset]["phase"] != "adjustment":
                raise ValueError("H30 crosses the adjustment boundary")
            values.append(r["action"])
        states.append(state)
        chunks.append(np.asarray(values, dtype=np.float32) - state)
    return np.asarray(states), np.asarray(chunks).reshape(-1, 7)


def rediscretize(row, ctx, stats):
    row = copy.deepcopy(row)
    if "history_sources" in row:
        points = []
        for source in row["history_sources"]:
            frame = ctx["frames"][int(source["global_index"])]
            if (frame["episode_id"], frame["attempt_id"], frame["frame_index"]) != (
                row["episode_id"],
                source["attempt_id"],
                source["frame_index"],
            ):
                raise ValueError("History crosses episode or has wrong physical identity")
            if frame["ros_timestamp"] > row["timestamp"] + 1e-4:
                raise ValueError("Future qpos in history")
            points.append(frame["observation.state"])
        if len(points) != 11 or int(row["history_sources"][-1]["global_index"]) != int(row["global_index"]):
            raise ValueError("History must be 11 points including current frame")
        row["qpos_h100_11_discrete"] = discretize_state_qpos(points, stats).tolist()
        prefix, separator, _ = row["prompt"].partition("State history H100 sampled to 11 points: ")
        if not separator:
            raise ValueError("Missing phase history text")
        row["prompt"] = prefix + separator + compact_history(row["qpos_h100_11_discrete"])
    row["prompt"] = canonical_touch(row["prompt"])
    return row


def phase_row(frame, ctx, timelines):
    """Raw continuous H100, same original source offsets as V7.7.2."""
    episode = int(frame["episode_id"])
    timeline, positions = timelines[episode]
    position = positions[int(frame["index"])]
    window = timeline[max(0, position - 99) : position + 1]
    effective = len(window)
    window = [timeline[0]] * (100 - len(window)) + window
    points = [window[i] for i in np.linspace(0, 99, 11, dtype=int)]
    return {
        "split": ctx["metas"][(episode, int(frame["attempt_id"]))]["split"],
        "global_index": int(frame["index"]),
        "episode_id": episode,
        "attempt_id": int(frame["attempt_id"]),
        "frame_index": int(frame["frame_index"]),
        "timestamp": float(frame["ros_timestamp"]),
        "need_recovery": False,
        "prompt": f"Mode: phase\nTask: {frame['instruction']}\nTouch: {frame['tactile_caption']}\n"
        f"Recovery plan: {frame['input_recovery_plan'] or 'none'}\nState history H100 sampled to 11 points: []",
        "history_sources": [
            {
                "attempt_id": int(p["attempt_id"]),
                "frame_index": int(p["frame_index"]),
                "global_index": int(p["index"]),
                "timestamp": float(p["ros_timestamp"]),
            }
            for p in points
        ],
        "effective_episode_history_length": effective,
    }


def blackbox_need(ctx, split, seed):
    positive, dropped = [], []
    for i in ctx["index"]["splits"][split]["need"]["manifest_row_indices"]:
        r = ctx["manifests"]["need"][i]
        if r["need_recovery"]:
            (positive if correct_rotation(r, "blackbox", ctx["metas"]) else dropped).append(r)
    timelines = defaultdict(list)
    for r in ctx["frames"].values():
        timelines[int(r["episode_id"])].append(r)
    timelines = {e: (sorted(v, key=lambda r: (r["ros_timestamp"], r["index"])), None) for e, v in timelines.items()}
    timelines = {e: (v, {int(r["index"]): p for p, r in enumerate(v)}) for e, (v, _) in timelines.items()}
    negatives = {s: [] for s in NEGATIVE_SOURCES}
    stops = {}
    for row in ctx["lookup"].values():
        if row["phase"] == "adjustment":
            key = (int(row["episode_id"]), int(row["attempt_id"]))
            stops[key] = max(stops.get(key, -1), int(row["frame_index"]))
    for frame in ctx["frames"].values():
        meta = ctx["metas"][(int(frame["episode_id"]), int(frame["attempt_id"]))]
        if meta["split"] != split:
            continue
        f = int(frame["frame_index"])
        if meta["result"] == "failure" and f < int(meta["shift_frame_index"]):
            source = NEGATIVE_SOURCES[0]
        elif meta["task"] == "one_success":
            source = NEGATIVE_SOURCES[1]
        elif meta["result"] == "success" and int(frame["attempt_id"]) == 2:
            if f <= stops[(int(frame["episode_id"]), int(frame["attempt_id"]))]:
                continue
            source = NEGATIVE_SOURCES[2]
        else:
            continue
        row = phase_row(frame, ctx, timelines)
        row["source"] = source
        row["pre_failure_hard_window"] = source == NEGATIVE_SOURCES[0] and f >= max(
            0, int(meta["shift_frame_index"]) - 30
        )
        negatives[source].append(row)
    allocation = allocate_balanced_without_replacement(
        {s: len(v) for s, v in negatives.items()}, 3 * len(positive), seed=seed
    )
    chosen = []
    for source, candidates in negatives.items():
        hard = [r for r in candidates if r["pre_failure_hard_window"]]
        if len(hard) > allocation[source]:
            raise ValueError("Negative quota cannot retain all pre-F30 hard negatives")
        chosen.extend(hard)
        chosen.extend(
            deterministic_uniform_select(
                [r for r in candidates if not r["pre_failure_hard_window"]],
                allocation[source] - len(hard),
                seed=seed,
                source=source,
            )
        )
    result = positive + chosen
    result.sort(key=lambda r: stable_priority(seed, "v12_1_need", r["global_index"], r.get("source")))
    return result, dropped


def build(output_dir=OUTPUT, *, source_paths=None, seed=42):
    """Build one immutable index for both stages; no checkpoint input is needed."""
    from openpi.shared import normalize

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use another --output-dir")
    ctxs, source_hashes = load_sources(source_paths or SOURCES)
    state_parts, action_parts = [], []
    for domain in DOMAINS:
        ctx = ctxs[domain]
        states, actions = action_delta_chunks(ctx, ctx["index"]["splits"]["train"]["action"]["indices"])
        state_parts.append(states)
        action_parts.append(actions)
    stats_values = {"state": weighted_statistics(state_parts), "actions": weighted_statistics(action_parts)}
    del state_parts, action_parts
    state_stats = StateQuantileStats(np.asarray(stats_values["state"]["q01"]), np.asarray(stats_values["state"]["q99"]))
    manifests = {task: [] for task in TASKS[1:]}
    splits = {s: {d: {} for d in DOMAINS} for s in SPLITS}
    domains, audit, counts = {}, [], {}
    for domain in DOMAINS:
        ctx = ctxs[domain]
        domains[domain] = {
            "dataset_dir": str(ctx["dataset"]),
            "source_index": str(Path((source_paths or SOURCES)[domain]).resolve()),
            "source_training_data_hash": ctx["index"]["training_data_hash"],
            "action_index_file": ctx["index"]["action_index_file"],
            "action_index_sha256": sha256_file(Path(ctx["index"]["action_index_file"])),
            "phase_manifest_file": ctx["action"]["source_files"]["action_phase_manifest"]["path"],
            "episode_splits": {s: sorted({e for (e, _), m in ctx["metas"].items() if m["split"] == s}) for s in SPLITS},
        }
        counts[domain] = {}
        for split in SPLITS:
            action_indices = list(ctx["index"]["splits"][split]["action"]["indices"])
            splits[split][domain]["action"] = {"indices": action_indices, "sample_count": len(action_indices)}
            counts[domain][split] = {"action": len(action_indices)}
            for task in TASKS[1:]:
                selected = [
                    ctx["manifests"][task][i] for i in ctx["index"]["splits"][split][task]["manifest_row_indices"]
                ]
                dropped = []
                if domain == "blackbox" and task == "need":
                    selected, dropped = blackbox_need(ctx, split, seed)
                elif domain == "blackbox" and task in ("failure", "plan"):
                    selected, dropped = (
                        [r for r in selected if correct_rotation(r, domain, ctx["metas"])],
                        [r for r in selected if not correct_rotation(r, domain, ctx["metas"])],
                    )
                for row in dropped:
                    audit.append(
                        {
                            "domain": domain,
                            "split": split,
                            "task": task,
                            **{k: row[k] for k in ("global_index", "episode_id", "attempt_id", "frame_index")},
                            "rotation": rotation(row["prompt"]),
                            "reason": "no_correct_current_rotation",
                        }
                    )
                positions = []
                source_positions = {id(r): i for i, r in enumerate(ctx["manifests"][task])}
                for old in selected:
                    row = rediscretize(old, ctx, state_stats)
                    row["domain"] = domain
                    row["schema_version"] = "v12_1_manifest_v1"
                    row["source_manifest_row_index"] = source_positions.get(id(old))
                    positions.append(len(manifests[task]))
                    manifests[task].append(row)
                if not positions:
                    raise ValueError(f"Empty stream after filtering: {domain}/{split}/{task}")
                splits[split][domain][task] = {
                    "manifest_row_indices": positions,
                    "global_indices": [int(r["global_index"]) for r in selected],
                    "sample_count": len(positions),
                }
                counts[domain][split][task] = len(positions)
                if task == "need":
                    counts[domain][split]["need_positive"] = sum(bool(r["need_recovery"]) for r in selected)
                    counts[domain][split]["need_negative"] = len(selected) - counts[domain][split]["need_positive"]
    output_dir.mkdir(parents=True)
    norm_dir = output_dir / "norm_stats"
    normalize.save(
        norm_dir,
        {k: normalize.NormStats(**{n: np.asarray(v) for n, v in values.items()}) for k, values in stats_values.items()},
    )
    artifacts = {}
    for task, rows in manifests.items():
        path = output_dir / f"{task}_manifest.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))
        artifacts[str(path)] = sha256_file(path)
    audit_path = output_dir / "filter_audit.jsonl"
    audit_path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in audit))
    artifacts[str(audit_path)] = sha256_file(audit_path)
    for p, payload in [
        (
            output_dir / "summary.json",
            {
                "counts": counts,
                "policy": POLICY,
                "filtered": dict(Counter(f"{r['domain']}/{r['split']}/{r['task']}" for r in audit)),
            },
        ),
        (
            norm_dir / "summary.json",
            {
                "policy": POLICY["normalization"],
                "train_only": True,
                "domain_weights": {d: 0.5 for d in DOMAINS},
                "action_space": "delta_joint_position",
                "norm_stats_sha256": sha256_file(norm_dir / "norm_stats.json"),
            },
        ),
    ]:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        artifacts[str(p)] = sha256_file(p)
    artifacts[str(norm_dir / "norm_stats.json")] = sha256_file(norm_dir / "norm_stats.json")
    index = seal(
        {
            "schema_version": "v12_1_joint_training_index_v1",
            "data_profile": DATA_PROFILE,
            "seed": seed,
            "policy": POLICY,
            "domains": domains,
            "splits": splits,
            "norm_stats_dir": str(norm_dir),
            "counts": counts,
            "source_hashes": source_hashes,
            "artifact_hashes": artifacts,
            **{f"{t}_manifest_file": str(output_dir / f"{t}_manifest.jsonl") for t in TASKS[1:]},
        }
    )
    index_path = output_dir / "v12_1_training_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    validate_index(index)
    return index


def validate_index(index, *, verify_files=True):
    if index.get("schema_version") != "v12_1_joint_training_index_v1" or index.get("data_profile") != DATA_PROFILE:
        raise ValueError("V12.1 index schema/profile mismatch")
    if index.get("policy") != POLICY or index["training_data_hash"] != sha256_json(
        {k: v for k, v in index.items() if k != "training_data_hash"}
    ):
        raise ValueError("V12.1 policy/data hash mismatch")
    if set(index["domains"]) != set(DOMAINS):
        raise ValueError("Both physical tasks are required")
    for domain in DOMAINS:
        source = index["domains"][domain]
        if Path(source["dataset_dir"]).resolve() != DATASETS[domain].resolve():
            raise ValueError("Unexpected source dataset")
        es = source["episode_splits"]
        if any(set(es[a]) & set(es[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
            raise ValueError("Physical episode leaks across splits")
        if domain == "blackbox" and set().union(*(set(es[s]) for s in SPLITS)) != set(range(261, 351)):
            raise ValueError("Old blackbox episodes are forbidden")
    if verify_files:
        for p, digest in (index["source_hashes"] | index["artifact_hashes"]).items():
            if sha256_file(Path(p)) != digest:
                raise ValueError(f"V12.1 source/artifact changed: {p}")
        rows = {t: load_jsonl(Path(index[f"{t}_manifest_file"])) for t in TASKS[1:]}
        stats = read_json(Path(index["norm_stats_dir"]) / "norm_stats.json")["norm_stats"]["state"]
        state_stats = StateQuantileStats(np.asarray(stats["q01"]), np.asarray(stats["q99"]))
        contexts = {}
        for domain in DOMAINS:
            source = index["domains"][domain]
            action = read_json(source["action_index_file"])
            metas = {
                (int(r["episode_id"]), int(r["attempt_id"])): r
                for r in read_json(Path(action["source_files"]["v4_training_index"]["path"]).parent / "profile.json")[
                    "attempts"
                ]
            }
            original = read_json(source["source_index"])
            contexts[domain] = {
                "frames": load_columns(source["dataset_dir"]),
                "metas": metas,
                "index": original,
                "selected": {
                    s: {t: set(original["splits"][s][t]["manifest_row_indices"]) for t in TASKS[1:]} for s in SPLITS
                },
                "manifests": {t: load_jsonl(Path(original[f"{t}_manifest_file"])) for t in TASKS[1:]},
            }
        for split in SPLITS:
            for domain in DOMAINS:
                ctx = contexts[domain]
                action = index["splits"][split][domain]["action"]
                if action["indices"] != ctx["index"]["splits"][split]["action"]["indices"] or action[
                    "sample_count"
                ] != len(action["indices"]):
                    raise ValueError("Action candidates must remain exactly the source Stage A pool")
                for task in TASKS[1:]:
                    stream = index["splits"][split][domain][task]
                    if (
                        len(stream["manifest_row_indices"]) != stream["sample_count"]
                        or len(stream["global_indices"]) != stream["sample_count"]
                    ):
                        raise ValueError("Stream count/identity mismatch")
                    selected = [rows[task][i] for i in stream["manifest_row_indices"]]
                    for row, g in zip(selected, stream["global_indices"], strict=True):
                        if (
                            row["domain"] != domain
                            or row["split"] != split
                            or row["global_index"] != g
                            or row["episode_id"] not in index["domains"][domain]["episode_splits"][split]
                        ):
                            raise ValueError("Manifest row identity/split mismatch")
                        if domain == "blackbox" and (
                            task in ("failure", "plan") or task == "need" and row["need_recovery"]
                        ):
                            if not correct_rotation(row, domain, ctx["metas"]):
                                raise ValueError("Filtered blackbox stream contains a none/opposite-rotation positive")
                        source_position = row["source_manifest_row_index"]
                        if source_position is not None:
                            if source_position not in ctx["selected"][split][task]:
                                raise ValueError("Row is not selected in the original task/split")
                            original = rediscretize(ctx["manifests"][task][source_position], ctx, state_stats)
                            if any(
                                row.get(k) != value
                                for k, value in original.items()
                                if k not in ("schema_version", "domain")
                            ):
                                raise ValueError("Labels/prompt/history differ from the normalized source row")
                        elif domain != "blackbox" or task != "need" or row["need_recovery"]:
                            raise ValueError("Only new blackbox need negatives may lack a source manifest row")
                        else:
                            frame = ctx["frames"][g]
                            meta = ctx["metas"][(row["episode_id"], row["attempt_id"])]
                            if row["source"] == NEGATIVE_SOURCES[0] and (
                                meta["result"] != "failure" or row["frame_index"] >= meta["shift_frame_index"]
                            ):
                                raise ValueError("Dropped positive must never be relabeled as a hard negative")
                            if row["source"] == NEGATIVE_SOURCES[1] and meta["task"] != "one_success":
                                raise ValueError("Invalid one-success negative")
                            checked = rediscretize(row, ctx, state_stats)
                            if (
                                checked["prompt"] != row["prompt"]
                                or checked["qpos_h100_11_discrete"] != row["qpos_h100_11_discrete"]
                            ):
                                raise ValueError("Negative history is not discretized with shared norm")
                            if canonical_touch(frame["tactile_caption"]) not in row["prompt"]:
                                raise ValueError("A real negative caption was changed")
                    if task == "need":
                        p = sum(bool(r["need_recovery"]) for r in selected)
                        if not p or len(selected) != 4 * p:
                            raise ValueError("Need requires exact 1:3 positive/negative support")


def artifact_identity(index, index_path):
    action_identity = {
        s: {
            "count": sum(index["splits"][s][d]["action"]["sample_count"] for d in DOMAINS),
            "sha256": sha256_json({d: index["splits"][s][d]["action"]["indices"] for d in DOMAINS}),
        }
        for s in SPLITS
    }
    action_identity["all"] = {
        "count": sum(v["count"] for v in action_identity.values()),
        "sha256": sha256_json(action_identity),
    }
    return {
        "data_profile": DATA_PROFILE,
        "prompt_profile": "phase_v2",
        "data_config_hash": sha256_json(POLICY),
        "training_data_hash": index["training_data_hash"],
        "action_frame_manifest_hash": sha256_json({d: index["domains"][d]["action_index_sha256"] for d in DOMAINS}),
        "action_indices_identity": action_identity,
        "index_sha256": sha256_file(Path(index_path)),
        "norm_stats_sha256": sha256_file(Path(index["norm_stats_dir"]) / "norm_stats.json"),
    }
