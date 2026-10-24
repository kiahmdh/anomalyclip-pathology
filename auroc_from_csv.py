"""
Recompute patch-level and slide-level AUROC/AP (+ recall/precision/specificity/F1
at an operating threshold) from a saved per-patch CSV (patch_name,slide,label,Ascore)
+ reference.csv. No GPU. Drops non-finite (NaN/inf) scores.

--exclude_class micro   drops every slide whose reference 'class' is micro (or any
                        listed class) from BOTH patch-level and slide-level metrics.
Recall needs a threshold; default = Youden-optimal per level, override with --threshold.
"""
import argparse
import os
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def load_reference(path):
    """slide -> 1/0 from 'type'; also slide -> class string (negative/micro/macro/itc)."""
    ref, ref_class = {}, {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        cols = [h.strip().lower() for h in header] if header else []
        name_idx = cols.index('image') if 'image' in cols else 0
        type_idx = cols.index('type') if 'type' in cols else 1
        class_idx = cols.index('class') if 'class' in cols else None
        for row in reader:
            if len(row) <= type_idx:
                continue
            key = os.path.splitext(row[name_idx].strip())[0].lower()
            t = row[type_idx].strip().lower()
            if t in ('tumor', 'tumour'):
                ref[key] = 1
            elif t == 'normal':
                ref[key] = 0
            if class_idx is not None and len(row) > class_idx:
                ref_class[key] = row[class_idx].strip().lower()
    return ref, ref_class


def operating_point(y, s, threshold=None):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if threshold is None:
        fpr, tpr, thr = roc_curve(y, s)
        threshold = float(thr[int(np.argmax(tpr - fpr))])
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float('nan')
    return threshold, recall, precision, specificity, f1, (tp, fp, fn, tn)


def print_operating_point(y, s, fixed_threshold):
    thr, rec, prec, spec, f1, (tp, fp, fn, tn) = operating_point(y, s, fixed_threshold)
    tag = "fixed" if fixed_threshold is not None else "Youden-opt"
    print(f"  threshold   = {thr:.4f} ({tag})")
    print(f"  recall      = {rec * 100:.2f}   (sensitivity, TP/(TP+FN))")
    print(f"  precision   = {prec * 100:.2f}")
    print(f"  specificity = {spec * 100:.2f}")
    print(f"  F1          = {f1 * 100:.2f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reference_csv", default=None)
    ap.add_argument("--topk_frac", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed decision threshold; default = Youden-optimal per level")
    ap.add_argument("--exclude_class", nargs="+", default=None,
                    help="drop slides whose reference 'class' is in this list (e.g. micro)")
    args = ap.parse_args()

    gt, pr, slides = [], [], []
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                score = float(row['Ascore'])
            except (ValueError, KeyError):
                score = float('nan')
            gt.append(int(row['label'])); pr.append(score); slides.append(row['slide'])
    gt = np.array(gt); pr = np.array(pr, dtype=float); slides = np.array(slides)

    finite = np.isfinite(pr)
    n_bad = int((~finite).sum())
    if n_bad:
        bad = sorted(set(slides[~finite].tolist()))
        print(f"WARNING: {n_bad} / {len(pr)} patches had NaN/inf scores; dropped.")
        print(f"  affected slides ({len(bad)}): {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    gt, pr, slides = gt[finite], pr[finite], slides[finite]

    ref, ref_class = ({}, {})
    if args.reference_csv:
        ref, ref_class = load_reference(args.reference_csv)

    if args.exclude_class:
        excl = {c.lower() for c in args.exclude_class}
        drop_slides = {k for k, c in ref_class.items() if c in excl}
        present = {s.lower() for s in set(slides.tolist())}
        dropped_present = sorted(drop_slides & present)
        keep = np.array([s.lower() not in drop_slides for s in slides])
        n_drop = int((~keep).sum())
        gt, pr, slides = gt[keep], pr[keep], slides[keep]
        print(f"EXCLUDED class={sorted(excl)}: {len(dropped_present)} slides in this test set, "
              f"{n_drop} patches removed")

    print(f"PATCH-LEVEL: n={len(gt)}  tumor={int((gt == 1).sum())}  normal={int((gt == 0).sum())}")
    if (gt == 1).any() and (gt == 0).any():
        print(f"  patch-AUROC = {roc_auc_score(gt, pr) * 100:.2f}")
        print(f"  patch-AP    = {average_precision_score(gt, pr) * 100:.2f}")
        print_operating_point(gt, pr, args.threshold)

    if args.reference_csv:
        from collections import defaultdict
        slide_scores = defaultdict(list)
        for s, sc in zip(slides, pr):
            slide_scores[s.lower()].append(float(sc))

        def top_frac_mean(scs):
            a = np.sort(np.asarray(scs, dtype=float))[::-1]
            k = max(1, int(np.ceil(len(a) * args.topk_frac)))
            return float(a[:k].mean())

        def report(agg_fn, name):
            sg, sp, missing = [], [], []
            for k, scs in slide_scores.items():
                if k in ref:
                    sg.append(ref[k]); sp.append(agg_fn(scs))
                else:
                    missing.append(k)
            sg, sp = np.array(sg), np.array(sp)
            print(f"SLIDE-LEVEL [{name}]: slides={len(sg)}  "
                  f"tumor={int((sg == 1).sum())}  normal={int((sg == 0).sum())}")
            if missing:
                print(f"  WARNING: {len(missing)} slides not in reference.csv, e.g. {missing[:5]}")
            if (sg == 1).any() and (sg == 0).any():
                print(f"  slide-AUROC = {roc_auc_score(sg, sp) * 100:.2f}")
                print(f"  slide-AP    = {average_precision_score(sg, sp) * 100:.2f}")
                print_operating_point(sg, sp, args.threshold)

        report(lambda s: float(np.max(s)), "max")
        report(top_frac_mean, f"top-{args.topk_frac * 100:g}%-mean")


if __name__ == '__main__':
    main()