from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageStat

from .dataset import discover_records, image_hash, validate_image


def audit(data_dir: str, mapping_csv: str | None = None, output_dir: str = "artifacts/audit") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    records = discover_records(data_dir, mapping_csv)
    checked = []
    hashes = defaultdict(list)
    for record in records:
        item = validate_image(record["path"]); item["label"] = record["label"]
        if item["readable"]:
            try: hashes[image_hash(record["path"])].append(record["path"])
            except OSError: pass
        checked.append(item)
    readable = [x for x in checked if x["readable"]]
    counts = Counter(x["label"] for x in readable)
    dimensions = Counter((x.get("width"), x.get("height")) for x in readable)
    quality = []
    for item in readable:
        with Image.open(item["path"]).convert("RGB") as image:
            stat = ImageStat.Stat(image)
            quality.append({"path": item["path"], "mean_brightness": sum(stat.mean) / 3, "mean_contrast": sum(stat.stddev) / 3})
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    summary = {
        "status": "complete" if records else "unavailable",
        "data_dir": str(Path(data_dir).resolve()),
        "total_discovered": len(records), "readable_images": len(readable),
        "unreadable_or_empty": len(checked) - len(readable),
        "classes": sorted(counts), "class_counts": dict(sorted(counts.items())),
        "class_distribution": {k: v / len(readable) for k, v in sorted(counts.items())} if readable else {},
        "duplicate_groups": duplicate_groups,
        "duplicate_image_count": sum(len(x) for x in duplicate_groups),
        "dimensions": {f"{k[0]}x{k[1]}": v for k, v in dimensions.items()},
        "quality": quality,
        "unsupported_extensions": sorted({Path(r["path"]).suffix.lower() for r in records if Path(r["path"]).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}),
        "limitations": ["Lighting, camera, background, and variety bias require manual review from the sample gallery.", "Duplicate files are detected by exact hash; perceptual similarity is not automatically treated as a duplicate."],
        "records": checked,
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if counts:
        plt.figure(figsize=(8, 4)); plt.bar(list(counts), list(counts.values()), color="#2f855a"); plt.xticks(rotation=30, ha="right"); plt.ylabel("Images"); plt.title("Discovered class distribution"); plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=150); plt.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", default="data/raw"); parser.add_argument("--mapping-csv"); parser.add_argument("--output-dir", default="artifacts/audit")
    args = parser.parse_args(); summary = audit(args.data_dir, args.mapping_csv, args.output_dir)
    print(json.dumps({k: summary[k] for k in ("status", "readable_images", "classes", "class_counts", "unreadable_or_empty")}, indent=2))


if __name__ == "__main__": main()
