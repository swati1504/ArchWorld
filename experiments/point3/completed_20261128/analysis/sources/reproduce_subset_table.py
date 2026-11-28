"""Reproduce the report's descriptive subset table; Python standard library only."""
from bisect import bisect_right
from collections import Counter
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def tv(first, second):
    a, b = Counter(first), Counter(second)
    return sum(abs(a[k] / len(first) - b[k] / len(second))
               for k in a.keys() | b.keys()) / 2


def ks(first, second):
    a, b = sorted(first), sorted(second)
    return max(abs(bisect_right(a, x) / len(a) - bisect_right(b, x) / len(b))
               for x in set(a + b))


def comparison():
    selection = load("point3_selection_summary.json")
    audit = load("point3_subset_audit.json")
    training = load("training_metadata.json")
    holdout = load("evaluation_landmarks.json")
    anchors = load("supplied_anchors.json")
    with (HERE / "ArchWorld_metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        full = {row["scene"]: row for row in csv.DictReader(f)}
    selected = selection["selected_landmarks"]
    assert len(selected) == len(set(selected)) == 54
    assert len(training) == 317 and len(holdout) == 79 and len(full) == 396
    assert set(selected) <= set(training) and not set(selected) & set(holdout)
    assert set(full) == set(training) | set(holdout)
    label = lambda s: anchors.get(s) or "__MISSING__"
    assert all(label(s) == row["anchor"] for s, row in training.items())
    result = {"selected_count": 54, "training_count": 317, "full_count": 396,
              "scope": "Post-selection metadata comparison; no held-out image descriptors or outcomes used.",
              "table": {}}
    for name, csv_key, training_key in [
        ("building_type_tv", "Architecture Type", "building_type"),
        ("region_tv", "Region", "region"),
    ]:
        part = [full[s][csv_key] for s in selected]
        train_value = tv(part, [row[training_key] for row in training.values()])
        assert abs(train_value - audit["metrics"][name]) < 1e-12
        result["table"][name] = {"training317": train_value,
                                "full396": tv(part, [row[csv_key] for row in full.values()])}
    result["table"]["anchor_tv"] = {
        "training317": tv([label(s) for s in selected], [label(s) for s in training]),
        "full396": tv([label(s) for s in selected], [label(s) for s in full]),
    }
    result["table"]["extent_m_ks"] = {
        "training317": ks([training[s]["extent_m"] for s in selected],
                          [r["extent_m"] for r in training.values()]),
        "full396": ks([float(full[s]["Extent"]) for s in selected],
                      [float(r["Extent"]) for r in full.values()]),
    }
    for name in ("image_count_ks", "duplicate_fraction_ks", "median_megapixels_ks"):
        result["table"][name] = {"training317": audit["metrics"][name], "full396": None}
    for name, row in result["table"].items():
        assert abs(row["training317"] - audit["metrics"][name]) < 1e-12
    result["anchor_categories"] = {
        "selected": len({label(s) for s in selected}),
        "training": len({label(s) for s in training}),
        "full": len({label(s) for s in full}),
        "absent_from_subset": sorted({label(s) for s in full} - {label(s) for s in selected}),
    }
    result["coverage_training_unselected"] = {
        key: audit["metrics"][key] for key in ("unselected_count", "coverage_mean", "coverage_p95")
    }
    return result


if __name__ == "__main__":
    print(json.dumps(comparison(), indent=2, ensure_ascii=True, allow_nan=False))
