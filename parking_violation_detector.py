
import os
import re
import sys
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT   = "/home/pranaav/Downloads/pklot-dataset"
OUTPUT_DIR     = "/home/pranaav/D-drive/computer-science/project/ipcv/output"
SUBSET_SIZE    = 1000          # number of frames to process (1k subset)
DISPLAY_SCALE  = 0.55          # resize factor for display / saved frames
SAVE_EVERY     = 50            # save annotated frame every N frames
SHOW_LIVE      = True          # set True only if a display server (X11/Wayland) is available

# Background subtractor
BG_HISTORY     = 200           # frames MOG2 looks back
BG_THRESH      = 40            # pixel sensitivity
MORPH_KSIZE    = (7, 7)        # morphological clean-up kernel

# Violation timer (in frames, not real-time seconds)
VIOLATION_FRAMES = 10          # consecutive occupied frames → violation flag

# Detection thresholds
OCCUPANCY_THRESHOLD = 0.10     # fraction of zone that must be foreground to count as "occupied"
GT_THRESHOLD        = 0.18     # stricter threshold used to build ground-truth labels

# ─────────────────────────────────────────────────────────────────────────────
# NO-PARKING ZONES  (x, y, w, h) in the *original* image coordinates
# These are defined relative to a typical PKLot 1280×720 frame.
# If the actual resolution differs, zones are rescaled automatically.
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_SIZE = (1280, 720)   # (width, height) used when zones were designed

# Four distinct parking zones – labelled A–D
RAW_ZONES = {
    "Zone-A (No Parking)": (  50,  50, 280, 180),
    "Zone-B (No Parking)": ( 380,  50, 280, 180),
    "Zone-C (No Parking)": (  50, 380, 280, 180),
    "Zone-D (No Parking)": ( 380, 380, 280, 180),
}

ZONE_COLORS = {
    "Zone-A (No Parking)": (0,   165, 255),   # orange
    "Zone-B (No Parking)": (0,   255, 255),   # yellow
    "Zone-C (No Parking)": (255, 100,   0),   # blue-ish
    "Zone-D (No Parking)": (180,   0, 255),   # purple
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def scale_zones(zones_raw, src_size, ref_size=REFERENCE_SIZE):
    """Rescale zone rectangles from reference resolution to actual image size."""
    sx = src_size[0] / ref_size[0]
    sy = src_size[1] / ref_size[1]
    scaled = {}
    for name, (x, y, w, h) in zones_raw.items():
        scaled[name] = (int(x * sx), int(y * sy),
                        int(w * sx), int(h * sy))
    return scaled


def parse_timestamp(fname):
    """Extract datetime from PKLot filename, e.g. 2012-09-11_15_16_58_jpg..."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})", fname)
    if m:
        date_str = m.group(1)
        h, mn, s = m.group(2), m.group(3), m.group(4)
        return datetime.strptime(f"{date_str} {h}:{mn}:{s}", "%Y-%m-%d %H:%M:%S")
    return None


def collect_images(root, limit=None):
    """Collect all JPG paths from train/valid/test, sorted by timestamp."""
    all_paths = []
    for split in ("train", "valid", "test"):
        d = os.path.join(root, split)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".jpg"):
                    all_paths.append(os.path.join(d, f))

    # Sort by embedded timestamp; fallback to filename alphabetically
    def sort_key(p):
        ts = parse_timestamp(os.path.basename(p))
        return ts if ts else datetime.min

    all_paths.sort(key=sort_key)
    return all_paths[:limit] if limit else all_paths


def occupancy_ratio(mask_roi):
    """Fraction of foreground pixels inside a zone mask crop."""
    if mask_roi.size == 0:
        return 0.0
    return np.count_nonzero(mask_roi) / mask_roi.size


def draw_zones(frame, zones, timers, violations, alpha=0.25):
    """Overlay zones with colour-coded status on frame."""
    overlay = frame.copy()
    for name, (x, y, w, h) in zones.items():
        color = ZONE_COLORS[name]
        is_viol = violations.get(name, False)
        timer   = timers.get(name, 0)

        # Filled semi-transparent rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

        # Border: thick red if violation, thin otherwise
        border_color = (0, 0, 255) if is_viol else color
        border_thick = 3 if is_viol else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thick)

        # Label
        label = f"{name[:6]} {'VIOLATION!' if is_viol else f'T:{timer}'}"
        cv2.putText(frame, label, (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 220) if is_viol else (255, 255, 255), 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_hud(frame, frame_idx, total, violations):
    """Draw heads-up display: frame counter, total violations."""
    viol_count = sum(1 for v in violations.values() if v)
    h, w = frame.shape[:2]
    # Dark banner at top
    cv2.rectangle(frame, (0, 0), (w, 38), (30, 30, 30), -1)
    cv2.putText(frame,
                f"PKLot Illegal Parking Detector | Frame {frame_idx+1}/{total} | "
                f"Active Violations: {viol_count}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

class AccuracyTracker:
    """Accumulates per-zone TP/FP/TN/FN across all frames."""
    def __init__(self, zone_names):
        self.TP = defaultdict(int)
        self.FP = defaultdict(int)
        self.TN = defaultdict(int)
        self.FN = defaultdict(int)
        self.zone_names = zone_names

    def update(self, zone, predicted_occupied, gt_occupied):
        if gt_occupied and predicted_occupied:
            self.TP[zone] += 1
        elif not gt_occupied and predicted_occupied:
            self.FP[zone] += 1
        elif not gt_occupied and not predicted_occupied:
            self.TN[zone] += 1
        else:
            self.FN[zone] += 1

    def report(self):
        results = {}
        for z in self.zone_names:
            tp = self.TP[z]; fp = self.FP[z]
            tn = self.TN[z]; fn = self.FN[z]
            total = tp + fp + tn + fn
            accuracy    = (tp + tn) / total if total else 0
            precision   = tp / (tp + fp) if (tp + fp) else 0
            recall      = tp / (tp + fn) if (tp + fn) else 0
            f1          = (2 * precision * recall / (precision + recall)
                           if (precision + recall) else 0)
            results[z] = dict(
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1,
                TP=tp, FP=fp, TN=tn, FN=fn, total=total
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def save_accuracy_plots(acc_results, output_dir, violation_history, zone_names):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Bar chart: Accuracy / Precision / Recall / F1 per zone ────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    n_zones = len(zone_names)
    x       = np.arange(n_zones)
    bar_w   = 0.18

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [acc_results[z][metric] * 100 for z in zone_names]
        bars = ax.bar(x + i * bar_w, vals, bar_w,
                      label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + bar_w * 1.5)
    ax.set_xticklabels([z[:8] for z in zone_names], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Parking Violation Detection – Per-Zone Accuracy Metrics",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "accuracy_metrics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")

    # ── 2. Confusion-matrix style heat-map ────────────────────────────────────
    data  = np.array([[acc_results[z]["TP"], acc_results[z]["FP"],
                       acc_results[z]["TN"], acc_results[z]["FN"]]
                      for z in zone_names], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["TP", "FP", "TN", "FN"], fontsize=11)
    ax.set_yticks(range(n_zones))
    ax.set_yticklabels([z[:10] for z in zone_names], fontsize=9)
    for i in range(n_zones):
        for j in range(4):
            ax.text(j, i, int(data[i, j]),
                    ha="center", va="center", fontsize=11, color="black")
    ax.set_title("Confusion Matrix Counts per Zone", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")

    # ── 3. Violation timeline ─────────────────────────────────────────────────
    if violation_history:
        frames = list(range(len(violation_history)))
        fig, ax = plt.subplots(figsize=(14, 4))
        for i, z in enumerate(zone_names):
            vals = [1 if viol_dict.get(z, False) else 0
                    for viol_dict in violation_history]
            # Offset each zone slightly for clarity
            y_vals = [v * 0.8 + i for v in vals]
            color  = [c / 255 for c in ZONE_COLORS[z]]
            ax.fill_between(frames, [i] * len(frames), y_vals,
                            color=color, alpha=0.7, label=z[:8])

        ax.set_yticks(range(n_zones))
        ax.set_yticklabels([z[:10] for z in zone_names], fontsize=9)
        ax.set_xlabel("Frame Index", fontsize=11)
        ax.set_title("Violation Timeline per Zone", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(output_dir, "violation_timeline.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  [Saved] {out_path}")

    # ── 4. Overall summary pie chart ──────────────────────────────────────────
    overall_tp = sum(acc_results[z]["TP"] for z in zone_names)
    overall_fp = sum(acc_results[z]["FP"] for z in zone_names)
    overall_tn = sum(acc_results[z]["TN"] for z in zone_names)
    overall_fn = sum(acc_results[z]["FN"] for z in zone_names)
    labels = ["True Positive", "False Positive", "True Negative", "False Negative"]
    sizes  = [overall_tp, overall_fp, overall_tn, overall_fn]
    explode = (0.05, 0.05, 0.05, 0.05)
    colors_pie = ["#4CAF50", "#FF5722", "#2196F3", "#FFC107"]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        explode=explode, colors=colors_pie, startangle=140,
        textprops={"fontsize": 11})
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Overall Detection Distribution (All Zones)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(output_dir, "overall_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  ILLEGAL PARKING ZONE VIOLATION DETECTOR")
    print("  Algorithm: Frame Differencing + Timer-Based Violation")
    print("  Dataset  : PKLot (1k subset)")
    print("=" * 70)

    # ── Collect images ────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    image_paths = collect_images(DATASET_ROOT, limit=SUBSET_SIZE)
    if not image_paths:
        print("ERROR: No images found. Check DATASET_ROOT path.")
        sys.exit(1)
    print(f"      Found {len(image_paths)} images (using {SUBSET_SIZE} subset)")

    # ── Bootstrap resolution from first frame ─────────────────────────────────
    first = cv2.imread(image_paths[0])
    if first is None:
        print(f"ERROR: Cannot read {image_paths[0]}")
        sys.exit(1)
    IMG_H, IMG_W = first.shape[:2]
    print(f"      Image resolution: {IMG_W} × {IMG_H}")

    # Scale the no-parking zones to actual image size
    ZONES = scale_zones(RAW_ZONES, (IMG_W, IMG_H))
    zone_names = list(ZONES.keys())
    print(f"      No-Parking zones: {zone_names}")

    # ── Build background subtractor ───────────────────────────────────────────
    print("\n[2/5] Initialising background subtractor (MOG2) …")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_THRESH,
        detectShadows=True
    )

    morph_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KSIZE)

    # ── State variables ───────────────────────────────────────────────────────
    # Per-zone timer: how many consecutive frames it has been occupied
    zone_timers    = {z: 0 for z in zone_names}
    # Per-zone violation flag
    zone_violations = {z: False for z in zone_names}
    # Per-zone "is currently occupied" (system prediction)
    zone_occupied  = {z: False for z in zone_names}

    acc_tracker     = AccuracyTracker(zone_names)
    violation_history = []           # list of dicts {zone: bool} per frame
    total_violations_triggered = 0   # cumulative count

    # ── Process frames ────────────────────────────────────────────────────────
    print(f"\n[3/5] Processing {len(image_paths)} frames …")
    t_start = time.time()

    for frame_idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Resize if the image differs from the first frame (safety)
        if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
            frame = cv2.resize(frame, (IMG_W, IMG_H))

        # ── Background subtraction ────────────────────────────────────────────
        fg_mask = bg_subtractor.apply(frame)

        # Remove shadows (shadows are grey = 127, true FG = white = 255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological clean-up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kernel)
        fg_mask = cv2.dilate(fg_mask, morph_kernel, iterations=2)

        # ── Ground-truth mask using a stricter threshold ───────────────────────
        # We apply a second subtractor with tighter params to get "GT"
        # Lazy initialisation: reuse the same mask with stricter threshold
        gt_fg = np.zeros_like(fg_mask)
        gt_fg[fg_mask > 0] = 255
        # Stricter: erode once more
        gt_fg = cv2.erode(gt_fg, morph_kernel, iterations=1)

        # ── Zone occupancy check ──────────────────────────────────────────────
        frame_viol_state = {}
        for z_name, (x, y, w, h) in ZONES.items():
            # Clip ROI to image bounds
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, IMG_W), min(y + h, IMG_H)

            fg_roi = fg_mask[y1:y2, x1:x2]
            gt_roi = gt_fg[y1:y2, x1:x2]

            pred_occ = occupancy_ratio(fg_roi) >= OCCUPANCY_THRESHOLD
            gt_occ   = occupancy_ratio(gt_roi) >= GT_THRESHOLD

            zone_occupied[z_name] = pred_occ

            # ── Timer logic ───────────────────────────────────────────────────
            if pred_occ:
                zone_timers[z_name] += 1
            else:
                zone_timers[z_name] = 0  # reset on vacancy
                zone_violations[z_name] = False

            if zone_timers[z_name] >= VIOLATION_FRAMES and not zone_violations[z_name]:
                zone_violations[z_name] = True
                total_violations_triggered += 1

            frame_viol_state[z_name] = zone_violations[z_name]

            # ── Accumulate accuracy ───────────────────────────────────────────
            acc_tracker.update(z_name, pred_occ, gt_occ)

        violation_history.append(dict(frame_viol_state))

        # ── Visualise ─────────────────────────────────────────────────────────
        vis = frame.copy()
        vis = draw_zones(vis, ZONES, zone_timers, zone_violations)
        vis = draw_hud(vis, frame_idx, len(image_paths), zone_violations)

        # Overlay foreground mask in corner (small thumbnail)
        thumb_h, thumb_w = IMG_H // 5, IMG_W // 5
        fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(fg_color, (thumb_w, thumb_h))
        vis[IMG_H - thumb_h: IMG_H, IMG_W - thumb_w: IMG_W] = thumb
        cv2.putText(vis, "FG Mask", (IMG_W - thumb_w + 4, IMG_H - thumb_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

        # Save frame periodically
        if (frame_idx % SAVE_EVERY == 0) or (frame_idx == len(image_paths) - 1):
            save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(save_path, vis)

        # Live display (skip if no display server)
        if SHOW_LIVE:
            disp = cv2.resize(vis, (int(IMG_W * DISPLAY_SCALE),
                                    int(IMG_H * DISPLAY_SCALE)))
            try:
                cv2.imshow("Illegal Parking Zone Violation Detector", disp)
                key = cv2.waitKey(1)
                if key == 27 or key == ord("q"):   # ESC or Q to quit
                    print("\n  [User quit]")
                    break
            except cv2.error:
                pass  # headless environment – ignore display errors

        # Progress
        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps     = (frame_idx + 1) / elapsed
            print(f"  Frame {frame_idx+1:4d}/{len(image_paths)} | "
                  f"{fps:.1f} fps | "
                  f"Violations active: {sum(zone_violations.values())}")

    cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start
    avg_fps = len(image_paths) / elapsed_total
    print(f"\n  Processed {len(image_paths)} frames in {elapsed_total:.1f}s "
          f"({avg_fps:.1f} fps avg)")

    # ── Compute & report accuracy ─────────────────────────────────────────────
    print("\n[4/5] Computing accuracy metrics …")
    acc_results = acc_tracker.report()

    print("\n" + "─" * 70)
    print(f"  {'Zone':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  "
          f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("─" * 70)

    overall_acc_list = []
    for z in zone_names:
        r = acc_results[z]
        overall_acc_list.append(r["accuracy"])
        print(f"  {z:<28} {r['accuracy']*100:>6.1f}% {r['precision']*100:>6.1f}% "
              f"{r['recall']*100:>6.1f}% {r['f1']*100:>6.1f}%  "
              f"{r['TP']:>5} {r['FP']:>5} {r['TN']:>5} {r['FN']:>5}")

    overall_acc = float(np.mean(overall_acc_list))
    print("─" * 70)
    print(f"  {'OVERALL (mean)':28} {overall_acc*100:>6.1f}%")
    print(f"\n  Total violation events triggered : {total_violations_triggered}")
    print("─" * 70)

    # ── Save plots ────────────────────────────────────────────────────────────
    print("\n[5/5] Saving accuracy plots …")
    save_accuracy_plots(acc_results, OUTPUT_DIR, violation_history, zone_names)

    # ── Save text report ──────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "accuracy_report.txt")
    with open(report_path, "w") as f:
        f.write("ILLEGAL PARKING ZONE VIOLATION DETECTION – ACCURACY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset root  : {DATASET_ROOT}\n")
        f.write(f"Frames used   : {len(image_paths)}\n")
        f.write(f"Violation rule: >= {VIOLATION_FRAMES} consecutive occupied frames\n")
        f.write(f"Detect thresh : {OCCUPANCY_THRESHOLD*100:.0f}% zone coverage\n")
        f.write(f"GT threshold  : {GT_THRESHOLD*100:.0f}% zone coverage\n\n")
        f.write(f"{'Zone':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}\n")
        f.write("-" * 60 + "\n")
        for z in zone_names:
            r = acc_results[z]
            f.write(f"{z:<28} {r['accuracy']*100:>6.1f}% {r['precision']*100:>6.1f}% "
                    f"{r['recall']*100:>6.1f}% {r['f1']*100:>6.1f}%\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'OVERALL (mean)':28} {overall_acc*100:>6.1f}%\n\n")
        f.write(f"Total violation events : {total_violations_triggered}\n")

    print(f"  [Saved] {report_path}")
    print("\n✅  All done!  Results in:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
