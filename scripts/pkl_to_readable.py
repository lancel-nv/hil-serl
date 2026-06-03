#!/usr/bin/env python3
"""Convert SERL demo pickle files to readable CSV files and PNG images."""

"""
python scripts/pkl_to_readable.py --input demo_data/example_ur_5_demos_2026-06-03_02-51-23.pkl

"""
import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - optional dependency
    iio = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert demo .pkl transitions to readable CSV files and images.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to demo .pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory. Default: sibling folder with same file stem.",
    )
    parser.add_argument(
        "--max-array-expand",
        type=int,
        default=32,
        help="Maximum flattened array length to expand into multiple CSV columns.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Only export CSV files; skip image export.",
    )
    return parser.parse_args()


def load_transitions(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level, got {type(data).__name__}")
    if data and not isinstance(data[0], dict):
        raise ValueError(f"Expected list[dict], got first item {type(data[0]).__name__}")
    return data


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_image_array(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.ndim == 3 and arr.shape[-1] in (1, 3, 4)


def normalize_image_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size > 0 and arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr


def save_png(path: Path, image: np.ndarray) -> None:
    if Image is not None:
        Image.fromarray(image).save(path)
        return

    if iio is not None:
        iio.imwrite(path, image)
        return

    raise RuntimeError(
        "No image backend found. Install pillow or imageio to export PNG images."
    )


def flatten_leaf(
    row: Dict[str, Any],
    key: str,
    value: Any,
    max_array_expand: int,
) -> None:
    value = to_python_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        row[key] = value
        return

    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        row[key] = str(value)
        return

    if arr.dtype == object:
        row[key] = json.dumps(arr.tolist(), ensure_ascii=False)
        return

    if arr.ndim == 0:
        row[key] = to_python_scalar(arr.item())
        return

    flat = arr.reshape(-1)
    if flat.size <= max_array_expand:
        for idx, item in enumerate(flat):
            row[f"{key}_{idx}"] = to_python_scalar(item)
        row[f"{key}_shape"] = json.dumps(list(arr.shape))
    else:
        row[f"{key}_shape"] = json.dumps(list(arr.shape))
        row[f"{key}_sample"] = json.dumps(flat[:max_array_expand].tolist())


def flatten_mapping(
    row: Dict[str, Any],
    prefix: str,
    mapping: Dict[str, Any],
    max_array_expand: int,
) -> None:
    for key, value in mapping.items():
        column_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flatten_mapping(row, column_key, value, max_array_expand)
        else:
            flatten_leaf(row, column_key, value, max_array_expand)


def export_csv(path: Path, rows: List[Dict[str, Any]], preferred_cols: List[str]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    all_cols = set()
    for row in rows:
        all_cols.update(row.keys())

    ordered_cols = [col for col in preferred_cols if col in all_cols]
    ordered_cols += sorted(all_cols - set(ordered_cols))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols)
        writer.writeheader()
        writer.writerows(rows)


def export_images_for_observation(
    output_dir: Path,
    transition_idx: int,
    side_prefix: str,
    observation: Dict[str, Any],
    row: Dict[str, Any],
) -> None:
    images_root = output_dir / "images" / side_prefix
    for key, value in observation.items():
        if not is_image_array(value):
            continue

        image_array = normalize_image_array(value)
        image_dir = images_root / key
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{transition_idx:06d}.png"
        save_png(image_path, image_array)
        row[f"{side_prefix}_{key}_image_path"] = str(image_path.relative_to(output_dir))
        row[f"{side_prefix}_{key}_image_shape"] = json.dumps(list(np.asarray(value).shape))


def convert_demo(
    input_path: Path,
    output_dir: Path,
    max_array_expand: int,
    export_images: bool,
) -> Tuple[int, int]:
    transitions = load_transitions(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    transition_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []
    episode_idx = 0
    step_in_episode = 0
    episode_start_idx = 0
    episode_reward = 0.0
    episode_success = False

    for idx, transition in enumerate(transitions):
        row: Dict[str, Any] = {
            "transition_idx": idx,
            "episode_idx": episode_idx,
            "step_in_episode": step_in_episode,
            "reward": to_python_scalar(transition.get("rewards")),
            "mask": to_python_scalar(transition.get("masks")),
            "done": bool(transition.get("dones", False)),
        }

        observations = transition.get("observations", {})
        if isinstance(observations, dict):
            if export_images:
                export_images_for_observation(output_dir, idx, "obs", observations, row)
            for key, value in observations.items():
                if is_image_array(value):
                    continue
                flatten_leaf(row, f"obs_{key}", value, max_array_expand)

        next_observations = transition.get("next_observations", {})
        if isinstance(next_observations, dict):
            if export_images:
                export_images_for_observation(
                    output_dir, idx, "next_obs", next_observations, row
                )
            for key, value in next_observations.items():
                if is_image_array(value):
                    continue
                flatten_leaf(row, f"next_obs_{key}", value, max_array_expand)

        actions = transition.get("actions")
        if actions is not None:
            flatten_leaf(row, "action", actions, max_array_expand)

        infos = transition.get("infos", {})
        if isinstance(infos, dict):
            flatten_mapping(row, "info", infos, max_array_expand)
            episode_success = episode_success or bool(infos.get("succeed", False))

        transition_rows.append(row)

        reward = float(transition.get("rewards", 0.0) or 0.0)
        episode_reward += reward
        is_done = bool(transition.get("dones", False))
        if is_done:
            episode_rows.append(
                {
                    "episode_idx": episode_idx,
                    "start_transition_idx": episode_start_idx,
                    "end_transition_idx": idx,
                    "length": step_in_episode + 1,
                    "total_reward": episode_reward,
                    "succeed": episode_success,
                }
            )
            episode_idx += 1
            step_in_episode = 0
            episode_start_idx = idx + 1
            episode_reward = 0.0
            episode_success = False
        else:
            step_in_episode += 1

    if transitions and episode_start_idx < len(transitions):
        episode_rows.append(
            {
                "episode_idx": episode_idx,
                "start_transition_idx": episode_start_idx,
                "end_transition_idx": len(transitions) - 1,
                "length": step_in_episode + 1,
                "total_reward": episode_reward,
                "succeed": episode_success,
            }
        )

    export_csv(
        output_dir / "transitions.csv",
        transition_rows,
        preferred_cols=[
            "transition_idx",
            "episode_idx",
            "step_in_episode",
            "reward",
            "mask",
            "done",
            "info_succeed",
            "info_left",
            "info_right",
        ],
    )
    export_csv(
        output_dir / "episodes.csv",
        episode_rows,
        preferred_cols=[
            "episode_idx",
            "start_transition_idx",
            "end_transition_idx",
            "length",
            "total_reward",
            "succeed",
        ],
    )

    summary = {
        "input_file": str(input_path),
        "num_transitions": len(transition_rows),
        "num_episodes": len(episode_rows),
        "output_dir": str(output_dir),
        "images_exported": export_images,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(transition_rows), len(episode_rows)


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".pkl":
        raise ValueError(f"Input file must be .pkl: {input_path}")

    default_output_dir = input_path.parent / input_path.stem
    output_dir = (args.output_dir or default_output_dir).expanduser().resolve()

    num_transitions, num_episodes = convert_demo(
        input_path=input_path,
        output_dir=output_dir,
        max_array_expand=args.max_array_expand,
        export_images=not args.no_images,
    )

    print(f"Converted file: {input_path}")
    print(f"Saved to: {output_dir}")
    print(f"Transitions: {num_transitions}")
    print(f"Episodes: {num_episodes}")
    print(f"CSV files: {output_dir / 'transitions.csv'}, {output_dir / 'episodes.csv'}")
    if not args.no_images:
        print(f"Images root: {output_dir / 'images'}")


if __name__ == "__main__":
    main()
