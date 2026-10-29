"""Numbers and tables for the report, the viva notes and the observation notes, all read from results/ files.

Used by report/build_report.py and by ml/summary_sections.py (so the hand-written observations in results/*/notes.md can
contain {{placeholders}} instead of typed numbers).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = {"a_coco": "A-coco", "a_ft": "A-ft", "a2_authors": "A2-authors", "b_rules": "B-rules", "c_mobilenet": "C-mobilenet"}
METHOD_TEXT = {
    "A-coco": "any phone found, COCO model", "A-ft": "any phone found, fine-tuned model",
    "A2-authors": "FPI-Det authors' logic on our boxes", "B-rules": "our geometric rules", "C-mobilenet": "MobileNetV2 on the whole image"}
ENHANCE_NAMES = {"none": "none", "global_he": "global HE", "clahe": "CLAHE", "clahe_smooth": "CLAHE + blur"}


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def f3(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def text_file(path: str) -> str:
    p = ROOT / path
    return p.read_text().strip() if p.exists() else ""


# --- values ------------------------------------------------------------------------------------------------------

def collect_base() -> tuple[dict[str, str], dict]:
    R = {"stats": read("results/dataset_stats.json"), "train": read("results/train/train_summary.json"),
         "det": read("results/eval/detection.json"), "cls": read("results/eval/classification.json"),
         "tune": read("results/eval/tuning_summary.json"), "split": read("results/eval/splits/summary.json"),
         "speed": read("results/eval/speed.json"), "app_speed": read("results/eval/app_speed.json")["runs"],
         "ll": read("results/lowlight/lowlight.json"), "mn": read("results/mobilenet/metrics.json"),
         "cfg": read("shared/rules.json")}
    v: dict[str, str] = {}
    add_methods(v, R)
    add_detection(v, R)
    add_training(v, R)
    add_lowlight(v, R)
    add_speed_and_config(v, R)
    add_tables(v, R)
    return v, R


NOTES = {"failure_notes": "results/eval/failure_notes.md", "lowlight_notes": "results/lowlight/notes.md",
         "mobilenet_notes": "results/mobilenet/notes.md"}


def collect() -> tuple[dict[str, str], dict]:
    """All values, including the observation notes written by hand (their {{placeholders}} filled from the results)."""
    v, R = collect_base()
    base = dict(v)
    for key, path in NOTES.items():
        v[key] = render(text_file(path), base, path)
    return v, R


def rendered_note(path: str) -> str:
    """One observation note with its placeholders filled in (used by ml/summary_sections.py)."""
    return render(text_file(path), collect_base()[0], path)


def add_methods(v: dict, R: dict) -> None:
    for slug, name in METHODS.items():
        m = R["cls"]["methods"][name]["metrics"]
        for key, short in (("accuracy", "acc"), ("precision", "prec"), ("recall", "rec"), ("f1", "f1"), ("specificity", "spec")):
            v[f"{slug}_{short}"] = f3(m[key])
    v["n_held"] = f"{R['cls']['images']:,}"
    h = R["split"]["held_out"]
    v["held_using_rate"] = pct(h["using_rate"])
    v["held_using"] = f"{h['using_phone']:,}"
    v["n_tune"] = f"{R['split']['tune']['images']:,}"
    v["n_held_plus_tune"] = f"{R['split']['tune']['images'] + R['split']['held_out']['images']:,}"
    v["tune_using_rate"] = pct(R["split"]["tune"]["using_rate"])
    v["tune_f1"] = f3(R["tune"]["best"]["f1"])
    v["tune_baseline_ft_conf"] = f"{R['tune']['baselines']['A-ft']['phone_conf']:g}"
    v["tune_baseline_coco_conf"] = f"{R['tune']['baselines']['A-coco']['phone_conf']:g}"
    fail = R["cls"]["failures"]
    n_fn = fail["false_negatives"]
    v["fp"], v["fn"] = str(fail["false_positives"]), str(n_fn)
    v["fn_no_phone"] = str(fail["false_negative_by_cause"]["no phone detected"])
    v["fn_no_phone_pct"] = pct(fail["false_negative_by_cause"]["no phone detected"] / n_fn, 0)
    v["fp_on_phone"], v["fp_on_call"] = str(fail["false_positive_by_state"]["ON_PHONE"]), str(fail["false_positive_by_state"]["ON_CALL"])
    v["mn_disagree"], v["mn_only_right"], v["rules_only_right"] = mobilenet_disagreement(R)


def mobilenet_disagreement(R: dict) -> tuple[str, str, str]:
    """Counts saved by ml/evaluate.py next to the MobileNet results."""
    d = read("results/mobilenet/disagreement.json")
    return str(d["disagreements"]), str(d["only_mobilenet_right"]), str(d["only_rules_right"])


def add_detection(v: dict, R: dict) -> None:
    d = R["det"]
    for key, label in (("yolo26n_coco", "coco"), ("fine_tuned", "ft")):
        p = d["phone"][key]
        v[f"det_phone_{label}_map50"], v[f"det_phone_{label}_map5095"] = f3(p["map50"]), f3(p["map50_95"])
        v[f"det_phone_{label}_iou"] = f3(p["mean_iou_of_true_positives"])
        v[f"det_phone_{label}_prec"], v[f"det_phone_{label}_rec"] = f3(p["precision"]), f3(p["recall"])
        v[f"det_phone_{label}_conf"] = f"{p['operating_conf']:g}"
        for size in ("small", "medium", "large"):
            v[f"det_size_{size}_{label}"] = f3(p["map50_by_size"][size]["map50"])
            v[f"det_size_{size}_n"] = str(p["map50_by_size"][size]["ground_truth_boxes"])
    h = d["head"]["fine_tuned"]
    v["det_head_map50"], v["det_head_map5095"], v["det_head_iou"] = f3(h["map50"]), f3(h["map50_95"]), f3(h["mean_iou_of_true_positives"])
    v["det_head_prec"], v["det_head_rec"], v["det_head_conf"] = f3(h["precision"]), f3(h["recall"]), f"{h['operating_conf']:g}"
    v["n_test"] = f"{d['images']:,}"
    t = R["train"]["final_val"]["per_class"]
    v["val_phone_map50"], v["val_head_map50"] = f3(t["phone"]["map50"]), f3(t["head"]["map50"])
    s = R["stats"]
    v["val_no_box_images"], v["test_no_box_images"] = str(s["val"]["images_without_boxes"]), str(s["test"]["images_without_boxes"])
    v["val_images"], v["test_images"], v["train_images"] = f"{s['val']['images']:,}", f"{s['test']['images']:,}", f"{s['train']['images']:,}"
    v["train_phone_boxes"], v["train_head_boxes"] = f"{s['train']['boxes']['phone']:,}", f"{s['train']['boxes']['head']:,}"
    nohead = s["usage_labels"]["phone_box_rate_per_label_csv_class"]
    v["phone_box_rate_using"] = pct(nohead["1"], 0)
    v["phone_box_rate_not_using"] = pct(nohead["0"], 0)


def add_training(v: dict, R: dict) -> None:
    t = R["train"]
    v["train_epochs"], v["train_hours"], v["train_gpu"] = str(t["epochs_run"]), f"{t['train_hours']:.1f}", t["gpu"]
    s = t["settings"]
    v["train_imgsz"], v["train_batch"], v["train_seed"] = str(s["imgsz"]), str(s["batch"]), str(s["seed"])
    a = t["augmentation"]
    v["train_aug"] = (f"mosaic {a['mosaic']:g}, horizontal flip {a['fliplr']:g}, scale ±{a['scale']:g}, translate {a['translate']:g}, "
                      f"colour jitter (HSV {a['hsv_h']:g}/{a['hsv_s']:g}/{a['hsv_v']:g}), random erasing {a['erasing']:g}")
    m = R["mn"]
    v["mn_train"], v["mn_val"], v["mn_ratio"] = str(m["train_images"]), str(m["val_images"]), f"{m['class_ratio']:.2f}"
    v["mn_epochs1"], v["mn_epochs2"] = str(m["stage1_epochs_run"]), str(m["stage2_epochs_run"])
    v["mn_unfrozen"], v["mn_threshold"] = str(m["stage2"]["unfrozen_layers"]), f"{m['threshold']:g}"
    h = read("results/mobilenet/history.json")
    v["mn_final_train_acc"], v["mn_final_val_acc"] = f3(h["accuracy"][-1]), f3(h["val_accuracy"][-1])
    cm = R["cls"]["methods"]
    v["mn_gap_points"] = f"{100 * (cm['B-rules']['metrics']['accuracy'] - cm['C-mobilenet']['metrics']['accuracy']):.0f}"


def add_lowlight(v: dict, R: dict) -> None:
    ll = R["ll"]
    cells = ll["test_map50"]
    for cond in ("original", "dim", "dark", "real_dark"):
        for method in ("none", "global_he", "clahe", "clahe_smooth"):
            v[f"ll_{cond}_{method}_phone"] = f3(cells[cond][method]["phone"])
            v[f"ll_{cond}_{method}_head"] = f3(cells[cond][method]["head"])
    p = ll["val_choice"]["params"]
    v["clahe_clip"], v["clahe_tiles"], v["smooth_ksize"] = f"{p['clip_limit']:g}", f"{p['tile_grid'][0]}×{p['tile_grid'][1]}", str(p["smooth_ksize"])
    v["app_threshold"], v["app_method"] = str(ll["app"]["low_light_threshold"]), ENHANCE_NAMES[ll["app"]["enhance_method"]]
    b = ll["brightness"]
    v["bright_orig_median"], v["bright_dim_median"], v["bright_dark_median"] = (f"{b[k]['p50']:.0f}" for k in ("original", "dim", "dark"))
    v["bright_real_dark_max"] = f"{b['real_dark']['max']:.0f}"
    v["n_real_dark"] = str(ll["test_counts"]["real_dark"])
    u = ll["usage_heldout"]["rows"]
    v["ll_usage_orig"], v["ll_usage_dark"] = f3(u["original/none"]["accuracy"]), f3(u["dark/none"]["accuracy"])
    v["val_sample"] = str(ll["val_choice"]["val_images"])
    v["dim_gain_phone"] = f"{100 * (cells['dim']['clahe_smooth']['phone'] - cells['dim']['none']['phone']):.1f}"


def add_speed_and_config(v: dict, R: dict) -> None:
    v["cpu"] = R["speed"]["cpu"]
    v["speed_ft_ms"] = f"{R['speed']['fine_tuned']['mean_ms']:.0f}"
    by = {str(r["imgsz"]): r for r in R["app_speed"]}
    v["app_ms_640"], v["app_fps_640"] = f"{by['640']['mean_ms']:.1f}", f"{by['640']['fps_mean']:.0f}"
    v["app_ms_480"], v["app_fps_480"] = f"{by['480']['mean_ms']:.1f}", f"{by['480']['fps_mean']:.0f}"
    c = R["cfg"]
    v["conf_phone"], v["conf_head"] = f"{c['conf']['phone']:g}", f"{c['conf']['head']:g}"
    v["head_expand"], v["max_dx"], v["max_dy"] = f"{c['call']['head_expand']:g}", f"{c['using']['max_dx_head_widths']:g}", f"{c['using']['max_dy_head_heights']:g}"
    sm = c["smoothing"]
    v["window_s"], v["enter_pct"], v["away_s"] = f"{sm['window_ms'] / 1000:g}", f"{100 * sm['enter_ratio']:.0f}%", f"{sm['away_ms'] / 1000:g}"
    v["lag_s"] = f"{sm['enter_ratio'] * sm['window_ms'] / 1000:.1f}"
    v["grid_size"] = str(sum(1 for _ in (ROOT / "results/eval/tuning.csv").open()) - 1)  # one row per combination tried


def add_tables(v: dict, R: dict) -> None:
    s, d, ll = R["stats"], R["det"], R["ll"]
    v["table_dataset"] = table(["Detector split", "Images", "Head boxes", "Phone boxes", "Images without any box"],
                               [[k, f"{s[k]['images']:,}", f"{s[k]['boxes']['head']:,}", f"{s[k]['boxes']['phone']:,}", str(s[k]["images_without_boxes"])]
                                for k in ("train", "val", "test")])
    sp = R["split"]
    v["table_splits"] = table(["Usage split", "Images", "Using a phone", "Share using"],
                              [[k.replace("_", "-"), f"{sp[k]['images']:,}", f"{sp[k]['using_phone']:,}", pct(sp[k]["using_rate"])] for k in ("tune", "held_out")])
    rows = [["COCO yolo26n (no fine-tuning)", "phone", f3(d["phone"]["yolo26n_coco"]["map50"]), f3(d["phone"]["yolo26n_coco"]["map50_95"]),
             f3(d["phone"]["yolo26n_coco"]["mean_iou_of_true_positives"]), f3(d["phone"]["yolo26n_coco"]["precision"]), f3(d["phone"]["yolo26n_coco"]["recall"])],
            ["Fine-tuned YOLO26n", "phone", f3(d["phone"]["fine_tuned"]["map50"]), f3(d["phone"]["fine_tuned"]["map50_95"]),
             f3(d["phone"]["fine_tuned"]["mean_iou_of_true_positives"]), f3(d["phone"]["fine_tuned"]["precision"]), f3(d["phone"]["fine_tuned"]["recall"])],
            ["Fine-tuned YOLO26n", "head", f3(d["head"]["fine_tuned"]["map50"]), f3(d["head"]["fine_tuned"]["map50_95"]),
             f3(d["head"]["fine_tuned"]["mean_iou_of_true_positives"]), f3(d["head"]["fine_tuned"]["precision"]), f3(d["head"]["fine_tuned"]["recall"])]]
    v["table_detection"] = table(["Model", "Class", "mAP50", "mAP50-95", "Mean IoU (true positives)", "Precision", "Recall"], rows)
    v["table_by_size"] = table(["Model", "Small (n=" + v["det_size_small_n"] + ")", "Medium (n=" + v["det_size_medium_n"] + ")", "Large (n=" + v["det_size_large_n"] + ")"],
                               [["COCO yolo26n"] + [v[f"det_size_{z}_coco"] for z in ("small", "medium", "large")],
                                ["Fine-tuned"] + [v[f"det_size_{z}_ft"] for z in ("small", "medium", "large")]])
    cm = R["cls"]["methods"]
    v["table_classification"] = table(["Method", "What it does", "Accuracy", "Precision", "Recall", "F1"],
                                      [[n, METHOD_TEXT[n]] + [f3(cm[n]["metrics"][k]) for k in ("accuracy", "precision", "recall", "f1")] for n in METHODS.values()])
    conds = {"original": "original", "dim": "dim", "dark": "dark", "real_dark": f"real dark photos ({ll['test_counts']['real_dark']})"}
    v["table_lowlight"] = table(["Lighting (test images)"] + [ENHANCE_NAMES[m] for m in ENHANCE_NAMES],
                                [[conds[c]] + [f"{f3(ll['test_map50'][c][m]['phone'])} / {f3(ll['test_map50'][c][m]['head'])}" for m in ENHANCE_NAMES] for c in conds])
    u = ll["usage_heldout"]["rows"]
    v["table_lowlight_usage"] = table(["Lighting / enhancement", "B-rules accuracy", "F1"],
                                      [[k.replace("/", " / ").replace("clahe_smooth", "CLAHE + blur").replace("none", "none"), f3(x["accuracy"]), f3(x["f1"])] for k, x in u.items()])
    v["table_speed"] = table(["Measurement", "Mean ms per image", "Frames per second"],
                             [["Detector alone, fine-tuned, 640 px (Phase 3)", f"{R['speed']['fine_tuned']['mean_ms']:.1f}", f"{1000 / R['speed']['fine_tuned']['mean_ms']:.0f}"]]
                             + [[f"Whole app pipeline, detector size {r['imgsz']}", f"{r['mean_ms']:.1f}", f"{r['fps_mean']:.0f}"] for r in R["app_speed"]])
    c = R["cfg"]
    v["table_rules"] = table(["Setting", "Value", "Meaning"],
                             [["conf.phone", v["conf_phone"], "lowest phone confidence kept"], ["conf.head", v["conf_head"], "lowest head confidence kept"],
                              ["call.head_expand", v["head_expand"], "head box grown by this fraction of its width and height (ON_CALL zone)"],
                              ["using.max_dx_head_widths", v["max_dx"], "ON_PHONE reach sideways, in head widths"],
                              ["using.max_dy_head_heights", v["max_dy"], "ON_PHONE reach downward, in head heights"],
                              ["smoothing.window_ms / enter_ratio", f"{c['smoothing']['window_ms']:g} / {c['smoothing']['enter_ratio']:g}", "a state is shown when it fills this share of the window"],
                              ["smoothing.away_ms", f"{c['smoothing']['away_ms']:g}", "no head for longer than this shows AWAY"]])
    v["readme_results"] = table(["Method (held-out images)", "Accuracy", "F1"], [[f"{n}: {METHOD_TEXT[n]}", f3(cm[n]["metrics"]["accuracy"]), f3(cm[n]["metrics"]["f1"])] for n in METHODS.values()]) \
        + f"\n\nPhone detection on the test split: mAP50 {v['det_phone_coco_map50']} (COCO) to {v['det_phone_ft_map50']} (fine-tuned). " \
          f"Processing speed of the app pipeline: {v['app_fps_640']} FPS on {v['cpu']} (without the camera). Full tables: `results/eval/summary.md` and `report/report.md`."



def render(template: str, values: dict[str, str], name: str) -> str:
    missing = sorted({k for k in re.findall(r"\{\{(\w+)\}\}", template) if k not in values})
    if missing:
        raise SystemExit(f"{name}: no value for placeholders {missing}. Regenerate the missing results or fix the template.")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template)
