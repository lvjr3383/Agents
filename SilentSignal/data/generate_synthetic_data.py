from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def _clamp_non_negative(value: float) -> int:
    return max(0, int(round(value)))


def generate_synthetic_dataset(
    output_csv: Path,
    media_manifest_path: Path,
    media_root: Path,
    seed: int = 42,
) -> dict[str, int]:
    random.seed(seed)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    media_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)

    center_lat = 29.9511  # New Orleans downtown
    center_lon = -90.0715
    offsets = [-0.03, -0.01, 0.01, 0.03]

    cells: list[dict[str, Any]] = []
    cell_index = 1
    for lat_off in offsets:
        for lon_off in offsets:
            cells.append(
                {
                    "cell_id": f"C-{cell_index:03d}",
                    "lat": round(center_lat + lat_off, 6),
                    "lon": round(center_lon + lon_off, 6),
                }
            )
            cell_index += 1

    primary_drop_cells = {"C-005", "C-006", "C-009", "C-010"}
    secondary_drop_cells = {"C-008", "C-012", "C-016"}

    start = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(16)]
    drop_window = set(timestamps[-4:])

    rows: list[dict[str, Any]] = []
    baseline_registry: dict[tuple[str, str], list[int]] = {}

    for cell in cells:
        for signal_type, baseline_center in (("cell", 58), ("social", 26)):
            key = (cell["cell_id"], signal_type)
            baseline_registry[key] = []

            for ts in timestamps:
                base_noise = random.uniform(-6.5, 6.5)
                baseline = baseline_center + base_noise

                if ts in drop_window and cell["cell_id"] in primary_drop_cells:
                    # Primary severe silence pocket.
                    if signal_type == "cell":
                        observed = random.uniform(0.0, 2.2)
                    else:
                        observed = random.uniform(0.0, 1.5)
                elif ts in drop_window and cell["cell_id"] in secondary_drop_cells:
                    # Secondary silence pocket with a milder drop profile.
                    if signal_type == "cell":
                        observed = random.uniform(7.0, 12.0)
                    else:
                        observed = random.uniform(2.0, 5.0)
                else:
                    observed = max(0.0, baseline + random.uniform(-5.0, 4.0))

                observed_int = _clamp_non_negative(observed)
                if ts not in drop_window:
                    baseline_registry[key].append(observed_int)

                rows.append(
                    {
                        "timestamp": _iso(ts),
                        "cell_id": cell["cell_id"],
                        "lat": cell["lat"],
                        "lon": cell["lon"],
                        "signal_type": signal_type,
                        "activity_count": observed_int,
                        "baseline_mean": 0,
                    }
                )

    baseline_lookup = {
        key: round(sum(values) / max(1, len(values)), 2)
        for key, values in baseline_registry.items()
    }

    for row in rows:
        row["baseline_mean"] = baseline_lookup[(row["cell_id"], row["signal_type"])]

    rows.sort(key=lambda row: (row["timestamp"], row["cell_id"], row["signal_type"]))

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "cell_id",
                "lat",
                "lon",
                "signal_type",
                "activity_count",
                "baseline_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    media_entries = [
        {
            "media_uri": "s3://silent-signal-media/media/drone_a_shout.mp4",
            "lat": 29.9411,
            "lon": -90.0815,
            "source": "drone-A",
            "duration_seconds": 42,
        },
        {
            "media_uri": "s3://silent-signal-media/media/drone_b_tap_pattern.mp4",
            "lat": 29.9611,
            "lon": -90.0615,
            "source": "drone-B",
            "duration_seconds": 55,
        },
        {
            "media_uri": "s3://silent-signal-media/media/drone_c_human_form.mp4",
            "lat": 29.9611,
            "lon": -90.0815,
            "source": "drone-C",
            "duration_seconds": 39,
        },
        {
            "media_uri": "s3://silent-signal-media/media/drone_d_ambient.mp4",
            "lat": 29.9811,
            "lon": -90.0415,
            "source": "drone-D",
            "duration_seconds": 61,
        },
    ]

    payload = {
        "seed": seed,
        "generated_at": _iso(datetime.now(timezone.utc)),
        "primary_drop_cells": sorted(primary_drop_cells),
        "secondary_drop_cells": sorted(secondary_drop_cells),
        "media": media_entries,
    }
    media_manifest_path.write_text(json.dumps(payload, indent=2))

    created_media = 0
    for entry in media_entries:
        relative_key = entry["media_uri"].split("/", 3)[-1]
        file_path = media_root / relative_key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            "Synthetic placeholder media asset for SilentSignal demo.\n"
            f"Source URI: {entry['media_uri']}\n"
        )
        created_media += 1

    return {"rows_written": len(rows), "media_assets": created_media}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SilentSignal synthetic data")
    parser.add_argument(
        "--output-csv",
        default="data/generated/signals.csv",
        help="Path to signals.csv output",
    )
    parser.add_argument(
        "--media-manifest",
        default="data/generated/media_manifest.json",
        help="Path to media manifest JSON",
    )
    parser.add_argument(
        "--media-root",
        default="data/s3_mock",
        help="Local root for mock S3 assets",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = generate_synthetic_dataset(
        output_csv=Path(args.output_csv),
        media_manifest_path=Path(args.media_manifest),
        media_root=Path(args.media_root),
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "signals_csv": args.output_csv,
                "media_manifest": args.media_manifest,
                "rows_written": summary["rows_written"],
                "media_assets": summary["media_assets"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
