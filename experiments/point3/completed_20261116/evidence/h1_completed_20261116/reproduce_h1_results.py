"""Verify the completed H1 snapshots and reproduce the milestone table/figure.

Run with Python and Matplotlib. This reads saved results; it does not run VGGT.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def csv_rows(name):
    with (HERE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def main():
    summary = load("run_summary.json")
    session = load("run_session_1_0.json")
    assert session["status"] == session["report_status"] == "complete"
    provenance = load("drive_provenance.json")
    for item in provenance["files"]:
        assert hashlib.sha256((HERE / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    reported = load("report_summary.json")
    study = load("run_study.json")
    plan = study["plan"]
    selected = load("run_selection.json")
    calibration = load("run_calibration_scores.json")
    rows = csv_rows("report_paired_results.csv")
    conditions = csv_rows("run_results.csv")
    assert summary["status"] == "complete" and summary["incomplete_landmarks"] == 0
    assert summary["confirmation_complete"] == len(rows) == 36
    assert summary["numeric_conditions_complete"] == len(conditions) == 198
    assert not csv_rows("run_failures.csv")
    assert plan["hypothesis"] == "H1" and selected["selection_uses_confirmation"] is False
    assert selected["plan_sha256"] == fingerprint(plan)
    assert selected["calibration_scores_sha256"] == fingerprint(calibration)
    calibration_ids = {r["scene"] for r in plan["rows"] if r["split"] == "calibration"}
    confirmation_ids = {r["scene"] for r in plan["rows"] if r["split"] == "confirmation"}
    assert len(calibration_ids) == 18 and len(confirmation_ids) == 36
    assert calibration_ids == set(calibration) == set(selected["calibration_scenes"])
    assert confirmation_ids == {r["scene"] for r in rows}
    assert not (calibration_ids & confirmation_ids)
    assert not ((calibration_ids | confirmation_ids) & set(plan["heldout_ids"]))
    candidate_names = sorted(next(iter(calibration.values())))
    calibration_means = {k: statistics.mean(v[k] for v in calibration.values())
                         for k in candidate_names}
    for prefix, key in [("scalar_", "global_scalar"), ("layer_", "H1")]:
        best = min((k for k in candidate_names if k.startswith(prefix)),
                   key=lambda k: (calibration_means[k], k))
        assert selected[key] == best
    indexed = {(r["scene"], r["condition"]): r for r in conditions}
    assert len(indexed) == 198
    assert len(plan["heldout_ids"]) == 79
    for scene, scores in calibration.items():
        assert set(scores) | {"no_intervention"} == {r["condition"] for r in conditions if r["scene"] == scene}
        for candidate, score in scores.items():
            original = indexed[(scene, candidate)]
            assert original["split"] == "calibration"
            assert math.isclose(score, float(original["CD_norm"]), rel_tol=1e-12)
    assert all(r["scene"] in calibration_ids | confirmation_ids for r in conditions)
    assert sum(r["split"] == "calibration" for r in conditions) == 90
    assert sum(r["split"] == "confirmation" for r in conditions) == 108
    for row in rows:
        assert row["candidate"] == selected["H1"] == "layer_alpha0.02"
        assert row["scalar_candidate"] == selected["global_scalar"] == "scalar_eta1000"
        for column, candidate in [("baseline", "no_intervention"),
                                  ("scalar", row["scalar_candidate"]),
                                  ("structured", row["candidate"])]:
            original = indexed[(row["scene"], candidate)]
            row[column] = float(row[column])
            assert math.isfinite(row[column])
            assert math.isclose(row[column], float(original["CD_norm"]), rel_tol=1e-12)
            assert int(row["num_views"]) == int(original["num_views"])
            assert original["split"] == "confirmation"
        for column, control in [("structured_minus_scalar", "scalar"),
                                ("structured_minus_baseline", "baseline")]:
            row[column] = float(row[column])
            assert math.isclose(row[column], row["structured"] - row[control], abs_tol=1e-12)
    differences = [r["structured_minus_scalar"] for r in rows]
    baseline_differences = [r["structured_minus_baseline"] for r in rows]
    metrics = {key: {"mean": statistics.mean(r[key] for r in rows),
                     "median": statistics.median(r[key] for r in rows)}
               for key in ["baseline", "scalar", "structured"]}
    paired = {"mean_difference": statistics.mean(differences),
              "median_difference": statistics.median(differences),
              "wins": sum(d < 0 for d in differences),
              "ties": sum(d == 0 for d in differences),
              "losses": sum(d > 0 for d in differences)}
    for key, value in paired.items():
        assert math.isclose(value, reported[key], abs_tol=1e-12)
        assert math.isclose(value, summary["hypotheses"]["H1"][key], abs_tol=1e-12)
    aggregate = {"status": "complete", "calibration_landmarks": 18,
                 "confirmation_landmarks": 36, "completed_conditions": 198,
                 "failures": 0, "selected_scalar": selected["global_scalar"],
                 "selected_layer_rule": selected["H1"], "calibration_means": calibration_means,
                 "metric": "extent-normalized Chamfer distance; equal landmark weights",
                 "confirmation_metrics": metrics, "structured_minus_scalar": paired,
                 "structured_minus_baseline": {"mean_difference": statistics.mean(baseline_differences),
                    "median_difference": statistics.median(baseline_differences),
                    "wins": sum(d < 0 for d in baseline_differences),
                    "ties": sum(d == 0 for d in baseline_differences),
                    "losses": sum(d > 0 for d in baseline_differences)},
                 "all_54_view_count": sum(summary["view_counts"].values()),
                 "confirmation_view_count": sum(int(r["num_views"]) for r in rows),
                 "study_source_sha256": study["source_sha256"],
                 "plan_sha256": selected["plan_sha256"],
                 "inference": "Descriptive single-seed internal confirmation; no significance or equivalence claim."}
    (HERE / "verified_statistics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42})
    ordered = sorted(rows, key=lambda r: r["structured_minus_scalar"])
    delta = [r["structured_minus_scalar"] for r in ordered]
    rank = list(range(1, len(delta) + 1))
    fig, ax = plt.subplots(figsize=(3.35, 2.12), layout="constrained")
    colors = ["#237e96" if d < 0 else "#b95324" for d in delta]
    ax.vlines(rank, 0, delta, colors=colors, linewidth=1.2, alpha=0.8)
    ax.scatter(rank, delta, c=colors, s=15, edgecolors="white", linewidths=0.35, zorder=3)
    ax.axhline(0, color="#343434", linewidth=0.7)
    ax.set_xlim(0, 37)
    ax.set_ylim(-13, 9)
    ax.set_xticks([1, 12, 24, 36])
    ax.set_yticks([-12, -6, 0, 6])
    ax.set_xlabel("Confirmation landmarks (ordered by difference)")
    ax.set_ylabel(r"$\Delta\,\mathrm{CD}_{\mathrm{norm}}$ (H1 $-$ scalar)")
    ax.text(0.025, 0.965, "23 improved / 13 worsened", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    figures = SOURCE / "figures"
    fig.savefig(figures / "point3_h1_confirmation.pdf")
    fig.savefig(figures / "point3_h1_confirmation.png", dpi=220)
    plt.close(fig)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
