"""Small, predefined 50-view studies. Selection uses metadata, not result quality."""
import copy

DATASET_REVISION = "5d4404323a21aa3de948f8bc57a053b29fa3d921"
SCENES = {
    "Akdamar_Church": ("arch", "West/Central Asia", "Church/Cathedral", 143.020),
    "Vank_Cathedral": ("arch", "West/Central Asia", "Church/Cathedral", 108.018),
    "Al-Azhar_Mosque": ("dome", "Africa", "Mosque", 71.510),
    "Ibrahim_Rouza": ("dome", "South Asia", "Tomb", 227.8),
    "Palácio_Nereu_Ramos": ("dome", "Latin America", "Public Building", 153.6),
    "Koutoubia_Mosque": ("minaret", "Africa", "Mosque", 98.094),
    "Great_Mosque_of_Testour": ("minaret", "Africa", "Mosque", 115.2),
    "New_Mosque_(Istanbul)": ("minaret", "West/Central Asia", "Mosque", 99.5),
}
SPLITS = {
    "H1": (["Akdamar_Church", "Al-Azhar_Mosque"], ["Vank_Cathedral", "Palácio_Nereu_Ramos"]),
    "H2": (["Al-Azhar_Mosque", "Koutoubia_Mosque"], ["Palácio_Nereu_Ramos", "New_Mosque_(Istanbul)"]),
    "H3": (["Al-Azhar_Mosque", "Ibrahim_Rouza", "Koutoubia_Mosque", "Great_Mosque_of_Testour"],
           ["Palácio_Nereu_Ramos", "New_Mosque_(Istanbul)"]),
}


def protocol(hypothesis):
    calibration, evaluation = SPLITS[hypothesis]
    rows = []
    for split, names in (("calibration", calibration), ("evaluation", evaluation)):
        for scene in names:
            anchor, region, category, mb = SCENES[scene]
            rows.append(dict(scene=scene, anchor=anchor, region=region, architecture_type=category,
                             archive_mb_approx=mb, split=split))
    return dict(hypothesis=hypothesis, num_views=50, seed=1729, eta_candidates=[100, 1000],
                alpha_candidates=[.01, .02], view_exponent=.5, view_min=.5, view_max=2.,
                min_calibration_per_anchor=2, max_step=100000., profile="uniform",
                dataset_revision=DATASET_REVISION, scenes=copy.deepcopy(rows),
                runtime_minutes=55, forecast_margin=1.3)


def variants(config):
    result = {f"scalar_eta{eta}": {"eta": eta, "config": {"mode": "fixed"}}
              for eta in config["eta_candidates"]}
    if config["hypothesis"] == "H1":
        for alpha in config["alpha_candidates"]:
            result[f"layer_alpha{alpha:g}"] = {"eta": 1, "config": {
                "mode": "layer_relative", "alpha": alpha, "profile": config["profile"],
                "max_step": config["max_step"]}}
    elif config["hypothesis"] == "H2":
        for eta in config["eta_candidates"]:
            result[f"view_eta{eta}"] = {"eta": eta, "config": {
                "mode": "view_fixed", "view_exponent": config["view_exponent"],
                "view_min": config["view_min"], "view_max": config["view_max"]}}
    return result


def job_count(config):
    calibration = sum(r["split"] == "calibration" for r in config["scenes"])
    evaluation = sum(r["split"] == "evaluation" for r in config["scenes"])
    return calibration * (1 + len(variants(config))) + evaluation * 3 + 1  # One reference check.


def select_settings(config, normalized_scores):
    """Read calibration scores only. Never accept evaluation scores for tuning."""
    calibration = [r for r in config["scenes"] if r["split"] == "calibration"]
    names = [r["scene"] for r in calibration]
    candidates = variants(config)
    if set(normalized_scores) != set(names):
        raise ValueError("Selection requires exactly the complete calibration scene set")
    if any(set(normalized_scores[s]) != set(candidates) for s in names):
        raise ValueError("Complete every candidate on every calibration scene before selecting")
    import math
    if any(not math.isfinite(v) for scores in normalized_scores.values() for v in scores.values()):
        raise ValueError("Calibration contains nonfinite scores")
    def best(prefix, scenes):
        eligible = [name for name in candidates if name.startswith(prefix)]
        return min(eligible, key=lambda k: (sum(normalized_scores[s][k] for s in scenes) / len(scenes),
                                            candidates[k]["eta"], candidates[k]["config"].get("alpha", 0)))
    scalar = best("scalar_", names)
    result = {"global_scalar": scalar, "calibration_scenes": names, "objective": "mean_CD_norm"}
    if config["hypothesis"] in ("H1", "H2"):
        result["structured"] = best("layer_" if config["hypothesis"] == "H1" else "view_", names)
    else:
        result["anchor_map"], result["anchor_counts"], result["fallback_types"] = {}, {}, []
        for anchor in sorted({r["anchor"] for r in config["scenes"]}):
            group = [r["scene"] for r in calibration if r["anchor"] == anchor]
            result["anchor_counts"][anchor] = len(group)
            supported = len(group) >= config["min_calibration_per_anchor"]
            result["anchor_map"][anchor] = best("scalar_", group) if supported else scalar
            if not supported:
                result["fallback_types"].append(anchor)
    return result
